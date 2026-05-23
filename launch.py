"""Графический модуль запуска и первичная настройка программного комплекса"""
from __future__ import annotations

import hashlib
import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Tk, Text, StringVar
from tkinter import ttk
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REQUIREMENTS = ROOT_DIR / "requirements.txt"
APP_FILE = ROOT_DIR / "app.py"
DEPS_MARKER = VENV_DIR / ".requirements.sha256"
DATA_SCRIPT = ROOT_DIR / "scripts" / "bootstrap_data.py"
LOCAL_URL = "http://localhost:8501"

IS_WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0


class LauncherApp:
    """Небольшое окно, которое показывает ход подготовки окружения и запуска локального сервера"""

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Прогнозирование численности населения - запуск")
        self.root.geometry("760x520")
        self.root.minsize(680, 460)
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_process: Optional[subprocess.Popen] = None
        self.streamlit_process: Optional[subprocess.Popen] = None

        self.title_var = StringVar(value="Подготовка проекта")
        self.status_var = StringVar(value="Проверка проекта")
        self.step_var = StringVar(value="")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        """Создаёт элементы интерфейса модуля запуска"""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#f8fafc")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Title.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 18, "bold"))
        style.configure("Text.TLabel", background="#ffffff", foreground="#334155", font=("Segoe UI", 10))
        style.configure("Small.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 9))
        style.configure("Accent.Horizontal.TProgressbar", troughcolor="#e2e8f0", background="#2563eb")
        style.configure("TButton", font=("Segoe UI", 10))

        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill=BOTH, expand=True)

        card = ttk.Frame(outer, style="Card.TFrame", padding=22)
        card.pack(fill=BOTH, expand=True)

        ttk.Label(card, textvariable=self.title_var, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            card,
            text="Модуль запуска проверяет виртуальное окружение Python, программные зависимости и локальные данные WPP. После подготовки веб-интерфейс открывается автоматически.",
            style="Text.TLabel",
            wraplength=680,
        ).pack(anchor="w", pady=(8, 18))

        ttk.Label(card, textvariable=self.status_var, style="Text.TLabel").pack(anchor="w")
        self.current_bar = ttk.Progressbar(card, mode="determinate", maximum=100, style="Accent.Horizontal.TProgressbar")
        self.current_bar.pack(fill=X, pady=(8, 16))

        ttk.Label(card, textvariable=self.step_var, style="Small.TLabel").pack(anchor="w")
        self.overall_bar = ttk.Progressbar(card, mode="determinate", maximum=100, style="Accent.Horizontal.TProgressbar")
        self.overall_bar.pack(fill=X, pady=(8, 16))

        self.log = Text(card, height=12, wrap="word", relief="flat", borderwidth=0, background="#f8fafc", foreground="#334155")
        self.log.pack(fill=BOTH, expand=True, pady=(4, 14))
        self.log.insert(END, "Журнал запуска\n")
        self.log.configure(state="disabled")

        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.pack(fill=X)
        self.open_button = ttk.Button(buttons, text="Открыть веб-интерфейс", command=lambda: webbrowser.open(LOCAL_URL))
        self.open_button.pack(side=LEFT)
        self.open_button.state(["disabled"])
        self.stop_button = ttk.Button(buttons, text="Остановить локальный сервер", command=self._stop_service)
        self.stop_button.pack(side=LEFT, padx=(10, 0))
        self.stop_button.state(["disabled"])
        self.close_button = ttk.Button(buttons, text="Закрыть окно", command=self._on_close)
        self.close_button.pack(side=RIGHT)

    def run(self) -> None:
        """Запускает рабочий поток и цикл обработки событий Tkinter"""
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(100, self._drain_messages)
        self.root.mainloop()

    def _on_close(self) -> None:
        """Останавливает локальный сервер Streamlit и закрывает окно модуля запуска"""
        self._stop_service()
        self.root.destroy()

    def _stop_service(self) -> None:
        """Останавливает процесс Streamlit, если он был запущен через модуль запуска"""
        process = self.streamlit_process
        if process is None or process.poll() is not None:
            self.streamlit_process = None
            self.stop_button.state(["disabled"])
            return
        self._post("status", "Остановка локального сервера Streamlit")
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        except Exception as exc:
            self._post("log", f"Не удалось остановить локальный сервер: {exc}")
        finally:
            self.streamlit_process = None
            self.stop_button.state(["disabled"])

    def _log(self, text: str) -> None:
        """Добавляет строку в журнал запуска"""
        self.log.configure(state="normal")
        self.log.insert(END, text.rstrip() + "\n")
        self.log.see(END)
        self.log.configure(state="disabled")

    def _post(self, kind: str, payload: object) -> None:
        """Передаёт сообщение из рабочего потока в интерфейс"""
        self.messages.put((kind, payload))

    def _drain_messages(self) -> None:
        """Обрабатывает сообщения рабочего потока и обновляет прогресс-бары"""
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(str(payload))
                self._log(str(payload))
            elif kind == "current":
                self.current_bar.configure(mode="determinate")
                self.current_bar["value"] = int(payload)
            elif kind == "overall":
                self.overall_bar["value"] = int(payload)
            elif kind == "step":
                self.step_var.set(str(payload))
            elif kind == "log":
                self._log(str(payload))
            elif kind == "done":
                self.title_var.set("Приложение запущено")
                self.status_var.set("Приложение доступно в браузере")
                self.current_bar["value"] = 100
                self.overall_bar["value"] = 100
                self.open_button.state(["!disabled"])
                self.stop_button.state(["!disabled"])
            elif kind == "error":
                self.title_var.set("Запуск остановлен")
                self.status_var.set(str(payload))
                self._log(str(payload))
                self.open_button.state(["disabled"])
        self.root.after(100, self._drain_messages)

    def _worker(self) -> None:
        """Выполняет проверку окружения, подготовку данных и запуск Streamlit"""
        steps = [
            ("Проверка виртуального окружения", self._ensure_venv),
            ("Установка зависимостей", self._ensure_requirements),
            ("Проверка и подготовка данных", self._ensure_data),
            ("Запуск локального сервера Streamlit", self._launch_streamlit),
        ]
        try:
            for index, (label, action) in enumerate(steps, start=1):
                self._post("step", f"Этап {index}/{len(steps)}: {label}")
                self._post("status", label)
                self._post("current", 0)
                action()
                self._post("current", 100)
                self._post("overall", int(index / len(steps) * 100))
            self._post("done", None)
        except Exception as exc:
            self._post("error", f"Ошибка: {exc}")

    def _venv_python(self) -> Path:
        """Возвращает путь к интерпретатору Python внутри .venv"""
        if IS_WINDOWS:
            return VENV_DIR / "Scripts" / "python.exe"
        return VENV_DIR / "bin" / "python"

    def _run_process(self, command: list[str], *, wait: bool = True, parse_data_progress: bool = False) -> Optional[subprocess.Popen]:
        """Запускает дочерний процесс без ручного взаимодействия с консолью"""
        self._post("log", "> " + " ".join(command))
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=CREATE_NO_WINDOW,
        )
        self.current_process = process
        if not wait:
            return process

        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if parse_data_progress and line.startswith("STATUS|"):
                self._post("status", line.split("|", 1)[1])
            elif parse_data_progress and line.startswith("PROGRESS|"):
                parts = line.split("|")
                if len(parts) >= 3:
                    value = int(float(parts[1]) / max(1, float(parts[2])) * 100)
                    self._post("current", value)
            elif parse_data_progress and line == "DONE":
                self._post("current", 100)
            elif parse_data_progress and line.startswith("ERROR|"):
                self._post("log", line.split("|", 1)[1])
            elif line:
                self._post("log", line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Команда завершилась с кодом {return_code}: {' '.join(command)}")
        return process

    def _ensure_venv(self) -> None:
        """Создаёт виртуальное окружение .venv, если оно ещё не подготовлено"""
        python_path = self._venv_python()
        if python_path.exists():
            self._post("status", "Виртуальное окружение найдено")
            self._post("current", 100)
            return
        self._post("status", "Создание .venv")
        self._run_process([sys.executable, "-m", "venv", str(VENV_DIR)])

    def _requirements_hash(self) -> str:
        """Считает хеш файла requirements.txt для контроля повторной установки зависимостей"""
        if not REQUIREMENTS.exists():
            return ""
        return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()

    def _ensure_requirements(self) -> None:
        """Устанавливает программные зависимости в .venv, если они ещё не соответствуют requirements.txt"""
        required_hash = self._requirements_hash()
        if DEPS_MARKER.exists() and DEPS_MARKER.read_text(encoding="utf-8") == required_hash:
            self._post("status", "Программные зависимости уже установлены")
            self._post("current", 100)
            return
        python_path = str(self._venv_python())
        self._post("status", "Обновление pip")
        self._run_process([python_path, "-m", "pip", "install", "--upgrade", "pip"])
        if REQUIREMENTS.exists():
            self._post("status", "Установка зависимостей проекта")
            self._run_process([python_path, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        DEPS_MARKER.write_text(required_hash, encoding="utf-8")

    def _ensure_data(self) -> None:
        """Запускает подготовку WPP-файла и локального CSV с историческими данными"""
        if not DATA_SCRIPT.exists():
            raise FileNotFoundError(f"Не найден скрипт подготовки данных: {DATA_SCRIPT}")
        self._run_process([str(self._venv_python()), str(DATA_SCRIPT)], parse_data_progress=True)

    def _launch_streamlit(self) -> None:
        """Запускает локальный сервер Streamlit в подготовленном окружении"""
        if not APP_FILE.exists():
            raise FileNotFoundError(f"Не найден файл приложения: {APP_FILE}")
        command = [
            str(self._venv_python()),
            "-m",
            "streamlit",
            "run",
            str(APP_FILE),
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
        self.streamlit_process = self._run_process(command, wait=False)
        self._post("status", "Ожидание запуска локального сервера")
        time.sleep(3)
        webbrowser.open(LOCAL_URL)


if __name__ == "__main__":
    LauncherApp().run()
