from __future__ import annotations

import io
import os
import sys
import re
import time
from pathlib import Path
from contextlib import redirect_stdout
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Import the simulator from the same directory or parent
# Tip: place qasm_gui.py next to quantum_simulator_global.py
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from quantum_simulator_global import QuantumSimulator, execute_qasm, _HAVE_CUDA
except Exception as e:
    messagebox.showerror("Import Error", f"quantum_simulator_global.py를 같은 폴더에 두세요.\n\n{e}")
    raise

APP_TITLE = "QASM Studio (Local)"
DEFAULT_QREG = 3

QASM_KEYWORDS = [
    # regs
    "qreg","creg",
    # 1q
    "i","id","x","y","z","h","s","sdg","t","tdg","rx","ry","rz","u3",
    # 2q/ctrl
    "swap","iswap","fsim","rxx","ryy","rzz",
    "cx","cy","cz","ch","cs","ct","cp","crx","cry","crz",
    "toffoli","ccx","cswap",
    # extended
    "u","u_full",
    # noise
    "noise_bitflip","nbf","noise_phaseflip","npf","noise_depolarizing","ndp","noise_amp","nad","noise_phase","nph",
    # util
    "measure","reset","seed","print_state","print_probs","print_creg","barrier","delay"
]

QASM_BUILTINS = set(QASM_KEYWORDS)

class QasmStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x720")
        self.minsize(900, 560)

        self.current_file: Path|None = None
        self.sim_seed = tk.IntVar(value=42)
        self.num_qubits = tk.IntVar(value=DEFAULT_QREG)
        self.gpu_enabled = tk.BooleanVar(value=bool(_HAVE_CUDA))

        self._build_ui()
        self._new_document(default_sample=True)

    # ---------------------- UI ----------------------
    def _build_ui(self):
        self._build_menu()
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Qubits:").pack(side="left")
        self.spin_qubits = tk.Spinbox(top, from_=1, to=30, textvariable=self.num_qubits, width=4)
        self.spin_qubits.pack(side="left", padx=(4, 16))

        ttk.Label(top, text="Seed:").pack(side="left")
        self.entry_seed = ttk.Entry(top, width=8, textvariable=self.sim_seed)
        self.entry_seed.pack(side="left", padx=(4, 16))

        ttk.Label(top, text="GPU:").pack(side="left")
        self.lbl_gpu = ttk.Label(top, text="사용 가능" if self.gpu_enabled.get() else "없음")
        self.lbl_gpu.pack(side="left", padx=(2, 16))

        self.btn_run = ttk.Button(top, text="실행 (F5)", command=self.run_qasm)
        self.btn_run.pack(side="left", padx=4)

        self.btn_clear = ttk.Button(top, text="출력 지우기", command=self.clear_output)
        self.btn_clear.pack(side="left", padx=4)

        # Panes
        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)

        # Editor
        left = ttk.Frame(panes)
        self.txt = tk.Text(left, wrap="none", undo=True, font=("Consolas", 12))
        self._attach_scrollbars(self.txt, parent=left)
        self._setup_highlight_tags(self.txt)
        self.txt.bind("<KeyRelease>", self._on_key_release)
        self.txt.bind("<Control-s>", self._save_shortcut)
        self.txt.bind("<F5>", lambda e: self.run_qasm())
        panes.add(left, weight=3)

        # Output
        right = ttk.Frame(panes)
        self.out = tk.Text(right, wrap="word", height=10, state="normal", font=("Consolas", 11))
        self._attach_scrollbars(self.out, parent=right)
        panes.add(right, weight=2)

        # Status bar
        self.status = ttk.Label(self, anchor="w", relief="sunken")
        self.status.pack(fill="x", side="bottom")
        self._set_status("준비됨")

    def _build_menu(self):
        m = tk.Menu(self)
        self.config(menu=m)

        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="새 파일", command=self._new_document, accelerator="Ctrl+N")
        fm.add_command(label="열기...", command=self._open_file, accelerator="Ctrl+O")
        fm.add_command(label="저장", command=self._save_file, accelerator="Ctrl+S")
        fm.add_command(label="다른 이름으로 저장...", command=self._save_as)
        fm.add_separator()
        fm.add_command(label="샘플 불러오기 - Bell", command=lambda: self._load_sample('bell'))
        fm.add_command(label="샘플 불러오기 - SYC", command=lambda: self._load_sample('syc'))
        fm.add_separator()
        fm.add_command(label="종료", command=self.destroy)
        m.add_cascade(label="파일", menu=fm)

        rm = tk.Menu(m, tearoff=0)
        rm.add_command(label="실행", command=self.run_qasm, accelerator="F5")
        rm.add_command(label="출력 지우기", command=self.clear_output)
        m.add_cascade(label="실행", menu=rm)

        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="정보", command=self._about)
        m.add_cascade(label="도움말", menu=hm)

        # accelerators
        self.bind("<Control-n>", lambda e: self._new_document())
        self.bind("<Control-o>", lambda e: self._open_file())
        self.bind("<Control-s>", self._save_shortcut)

    def _about(self):
        messagebox.showinfo("정보", "QASM Studio (Local)\n- quantum_simulator_global.py 기반\n- F5로 실행\n- QASM 파일 열기/저장 지원")

    def _attach_scrollbars(self, text_widget: tk.Text, parent: ttk.Frame):
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=text_widget.xview)
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=text_widget.yview)
        text_widget.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

    # ---------------------- Syntax highlight ----------------------
    def _setup_highlight_tags(self, txt: tk.Text):
        txt.tag_configure("kw", foreground="#0057b8")
        txt.tag_configure("num", foreground="#9b870c")
        txt.tag_configure("com", foreground="#888888")
        txt.tag_configure("err", background="#ffcccc")

    def _on_key_release(self, event=None):
        self._highlight_all()

    def _highlight_all(self):
        txt = self.txt
        content = txt.get("1.0", "end-1c")
        # clear tags
        for tag in ("kw","num","com","err"):
            txt.tag_remove(tag, "1.0", "end")

        # simple token highlight
        # comments
        for m in re.finditer(r"(#.*?$|//.*?$)", content, flags=re.M):
            s, e = m.span()
            self._tag_range(txt, s, e, "com")
        # numbers (floats, complex like 1+2i or 1+2j)
        for m in re.finditer(r"(?<![\w.])(?:[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:[+-]\d+(?:\.\d+)?[ij])?)", content):
            self._tag_range(txt, *m.span(), "num")
        # keywords
        kws = "|".join(sorted(map(re.escape, QASM_KEYWORDS), key=len, reverse=True))
        for m in re.finditer(rf"\b(?:{kws})\b", content, flags=re.I):
            self._tag_range(txt, *m.span(), "kw")

    def _idx_to_index(self, s: str, idx: int) -> str:
        # convert absolute char idx to Tk index "line.col"
        lines = s.splitlines(keepends=True)
        i = idx
        row = 1
        for line in lines:
            if i <= len(line)-1:
                col = i
                return f"{row}.{col}"
            i -= len(line)
            row += 1
        return f"{row}.0"

    def _tag_range(self, txt: tk.Text, start_idx: int, end_idx: int, tag: str):
        content = txt.get("1.0", "end-1c")
        s_index = self._idx_to_index(content, start_idx)
        e_index = self._idx_to_index(content, end_idx)
        txt.tag_add(tag, s_index, e_index)

    # ---------------------- File ops ----------------------
    def _new_document(self, default_sample: bool=False):
        self.txt.delete("1.0", "end")
        self.current_file = None
        if default_sample:
            self.txt.insert("1.0", "qreg 3\nh 0\ncx 0 1\nprint_state\n")
        self._highlight_all()
        self._set_status("새 문서")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="QASM 파일 열기",
            filetypes=[("QASM files","*.qasm"),("Text files","*.txt"),("All files","*.*")]
        )
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", text)
            self.current_file = Path(path)
            self._highlight_all()
            self._set_status(f"열었음: {self.current_file.name}")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _save_shortcut(self, event=None):
        self._save_file()
        return "break"

    def _save_file(self):
        if self.current_file is None:
            return self._save_as()
        try:
            with open(self.current_file, "w", encoding="utf-8") as f:
                f.write(self.txt.get("1.0", "end-1c"))
            self._set_status(f"저장됨: {self.current_file.name}")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _save_as(self):
        path = filedialog.asksaveasfilename(
            title="다른 이름으로 저장",
            defaultextension=".qasm",
            filetypes=[("QASM files","*.qasm"),("Text files","*.txt"),("All files","*.*")]
        )
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.txt.get("1.0", "end-1c"))
            self.current_file = Path(path)
            self._set_status(f"저장됨: {self.current_file.name}")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _load_sample(self, which: str):
        if which == 'bell':
            sample = "qreg 2\nh 0\ncx 0 1\nprint_state\n"
        elif which == 'syc':
            sample = "qreg 3\nh 0\ncx 0 1\nfsim 1 2 1.57079632679 0.5235987756\nprint_state\n"
        else:
            sample = "qreg 3\nh 0\ncx 0 1\nprint_state\n"
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", sample)
        self._highlight_all()
        self._set_status("샘플 불러옴")

    # ---------------------- Run ----------------------
    def clear_output(self):
        self.out.delete("1.0", "end")

    def _append_out(self, text: str):
        self.out.insert("end", text)
        self.out.see("end")

    def _detect_qreg(self, text: str) -> int|None:
        # return first qreg N if present
        m = re.search(r"\bqreg\s+(\d+)\b", text)
        if m:
            try: return int(m.group(1))
            except: return None
        return None

    def run_qasm(self):
        code = self.txt.get("1.0", "end-1c")
        if not code.strip():
            return

        # Decide qubit count: from qreg or from spinbox
        q_from_code = self._detect_qreg(code)
        n = q_from_code if q_from_code is not None else int(self.num_qubits.get())

        try:
            sim = QuantumSimulator(n, seed=int(self.sim_seed.get()))
        except Exception as e:
            messagebox.showerror("시뮬레이터 오류", str(e))
            return

        lines = [ln for ln in code.splitlines() if ln.strip()]
        buf = io.StringIO()
        t0 = time.time()
        try:
            with redirect_stdout(buf):
                execute_qasm(sim, lines=lines)
        except Exception as e:
            self._append_out(f"\n[오류] {e}\n")
            return
        dt = time.time() - t0

        output = buf.getvalue()
        self._append_out(f"\n===== 실행 완료 ({dt*1000:.1f} ms, qubits={n}, GPU={'Yes' if _HAVE_CUDA else 'No'}) =====\n")
        self._append_out(output + "\n")
        self._set_status("실행 완료")

    # ---------------------- Misc ----------------------
    def _set_status(self, msg: str):
        self.status.config(text=f" {msg}")

def main():
    app = QasmStudio()
    app.mainloop()

if __name__ == "__main__":
    main()
