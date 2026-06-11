import matplotlib
matplotlib.use('Agg')

import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
import threading
import queue
import sys
import os
import ast
import json
import time
import random
import requests
import numpy as np

import funktionen_mit_referenzen as fmr
from funktionen_mit_referenzen import (
    Oracle, learn_lexicon, grammar_generate, grammar_parse, explain_generate,
    StopLearning, UndoLearning, Vocabulary
)

# ── thread-safe communication ───────────────────────────────────────────────
_input_queue   = queue.Queue()   # GUI  →  algorithm  : "ZAHL WORT" | "STOP" | "UNDO"
_waiting_event = threading.Event()  # set while algorithm is blocked on input
_start_event   = threading.Event()  # set by GUI when oracle choice is locked (first word)

def _gui_input_func(prompt):
    """Replaces input() in hybrid2 branch. Blocks until GUI delivers a response."""
    _waiting_event.set()
    result = _input_queue.get()
    _waiting_event.clear()
    return result

fmr._hybrid2_input_func = _gui_input_func


# ── same oracle / cache setup as test.py ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_CACHE_PATH = os.path.join(BASE_DIR, 'search_cache.json')

def load_search_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warnung: Such-Cache nicht ladbar ({e}).")
        return {}

def save_search_cache(cache, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

search_cache = load_search_cache(SEARCH_CACHE_PATH)

def openlibrary_search_count(word, retries=3):
    from urllib.parse import quote
    url = "https://openlibrary.org/search/inside.json?q=%22" + quote(word) + "%22&limit=1"
    for attempt in range(retries):
        try:
            session = requests.Session()
            res = session.get(url, timeout=15)
            session.close()
            res.raise_for_status()
            data = res.json()
            amount = data.get('hits', {}).get('total', 0)
            print(word, amount)
            time.sleep(random.randint(3, 6))
            return amount
        except Exception as e:
            print(word, f"Versuch {attempt+1}/{retries} Fehler: {e}")
            if attempt < retries - 1:
                time.sleep(10)
    print(f"\n⚠️  WARNUNG: Verbindungsfehler bei '{word}' nach {retries} Versuchen — gebe 0 zurück!\n")
    return 0

def cached_openlibrary_search_count(word, retries=3):
    if word in search_cache:
        print(word, search_cache[word], "(cache)")
        return search_cache[word]
    result = openlibrary_search_count(word, retries=retries)
    search_cache[word] = result
    if len(search_cache) % 50 == 0:
        save_search_cache(search_cache, SEARCH_CACHE_PATH)
    return result


# ── oracle mode (set by toggle before algorithm starts) ─────────────────────
_oracle_style = 'hybrid3'   # default: no internet


# ── GUI ──────────────────────────────────────────────────────────────────────
COLS = 10
N_MAX = 999

COLOR_UNKNOWN  = '#f0f0f0'
COLOR_ABSTRACT = '#ffe066'   # yellow       – generierbar, aber nur durch Abstraktion
COLOR_PENDING  = '#ffa040'   # orange       – vom Bediener eingegeben, noch nicht confirmed
COLOR_FORCED   = '#b8e8b8'   # hellgrün     – confirmed, aber nicht direkt vom Bediener eingegeben
COLOR_KNOWN    = '#7ec87e'   # dunkelgrün   – direkt vom Bediener bestätigt
COLOR_WAITING  = '#d0e8ff'   # blue tint while algorithm is waiting

class _Tooltip:
    """Simple mouse-over tooltip for any tkinter widget."""
    def __init__(self, widget, text):
        self._widget = widget
        self._text   = text
        self._tip    = None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)

    def _show(self, _event=None):
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f'+{x}+{y}')
        tk.Label(
            self._tip,
            text=self._text,
            background='#ffffe0',
            relief=tk.SOLID,
            borderwidth=1,
            font=('Arial', 8),
            wraplength=280,
            justify=tk.LEFT,
            padx=4, pady=3,
        ).pack()

    def _hide(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


class ZahlwortGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Zahlwort-Lernmaschine")
        self.root.resizable(True, True)

        self._oracle_locked   = False   # becomes True after the first word is submitted
        self._owner          = {}        # number → root key of generating function
        self._func_line_map  = {}        # root key → 1-based line in _func_list
        self._line_func_map  = {}        # 1-based line → root key
        self._last_lex_keys  = frozenset()  # fingerprint: frozenset of (key, confirmed_maximum)
        self._pinned_root    = None      # root pinned by click
        self._hover_root     = None      # root highlighted by hover
        self._q_labels       = {}        # number → '?' label widget
        self._pending        = set()   # numbers submitted by operator, not yet confirmed
        self._pending_words  = {}      # number -> word submitted by operator (while pending)
        self._confirmed     = set()   # numbers with a confirmed output (any source)
        self._user_submitted = set()  # numbers directly entered/confirmed by the operator
        self._abstract      = set()   # numbers reachable only via abstraction
        self._words         = {}      # number -> last displayed word

        self._build_ui()
        self._check_cycle()     # start periodic update loop

    # ── layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Two stacked "pages": the learning grid and the grammar-test page.
        self._learn_page = tk.Frame(self.root)
        self._test_page  = tk.Frame(self.root)
        self._learn_page.pack(fill=tk.BOTH, expand=True)

        # --- status bar ---
        top = tk.Frame(self._learn_page)
        top.pack(fill=tk.X, padx=6, pady=4)

        self._status_var = tk.StringVar(value="Algorithmus läuft …")
        tk.Label(top, textvariable=self._status_var,
                 font=('Arial', 13, 'bold'), fg='#1a4a8a',
                 anchor='w').pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Button(top, text="TEST", bg='#2a7ab0', fg='white', width=7,
                  command=self._show_test_page).pack(side=tk.RIGHT, padx=4)
        tk.Button(top, text="UNDO", bg='#e07020', fg='white', width=7,
                  command=self._send_undo).pack(side=tk.RIGHT, padx=4)

        self._stat_oracle_var = tk.BooleanVar(value=False)
        self._oracle_cb = tk.Checkbutton(
            top,
            text="Statistisches Orakel (benötigt Internet-Verbindung)",
            variable=self._stat_oracle_var,
            font=('Arial', 9),
            command=self._on_oracle_toggle,
        )
        self._oracle_cb.pack(side=tk.RIGHT, padx=(0, 12))
        _Tooltip(
            self._oracle_cb,
            "Kann vor Beginn des Lernens eingestellt werden.\n"
            "Nach dem Start ist diese Einstellung gesperrt.",
        )

        # --- legend ---
        legend_frame = tk.Frame(self._learn_page, bg='#f5f5f5')
        legend_frame.pack(fill=tk.X, padx=6, pady=(0, 4))
        legend_items = [
            (COLOR_KNOWN,    'Gelehrte Wörter (direkt bestätigt)'),
            (COLOR_FORCED,   'Ausgedacht, muss nach Zwangsbedingung richtig sein'),
            (COLOR_ABSTRACT, 'Ausgedacht, durch statistisches Orakel bestätigt'),
            (COLOR_PENDING,  'Eingegeben, noch nicht verarbeitet'),
            (COLOR_UNKNOWN,  'Unbekannt'),
        ]
        for color, label in legend_items:
            box = tk.Frame(legend_frame, bg=color, width=16, height=16,
                           relief=tk.RAISED, bd=1)
            box.pack(side=tk.LEFT, padx=(6, 2), pady=2)
            box.pack_propagate(False)
            tk.Label(legend_frame, text=label, bg='#f5f5f5',
                     font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 10))
        # --- main area: grid left, function list right (resizable via sash) ---
        main_pane = tk.PanedWindow(self._learn_page, orient=tk.HORIZONTAL,
                                   sashrelief=tk.RAISED, sashwidth=6)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=4)
        self._main_pane = main_pane

        grid_frame = tk.Frame(main_pane)
        main_pane.add(grid_frame, stretch='always', minsize=100)

        # --- function list panel (right) ---
        func_frame = tk.LabelFrame(main_pane, text="Gelernte Funktionen", padx=4, pady=4)
        main_pane.add(func_frame, stretch='never', minsize=60)
        func_vsb = tk.Scrollbar(func_frame, orient=tk.VERTICAL)
        func_hsb = tk.Scrollbar(func_frame, orient=tk.HORIZONTAL)
        self._func_list = tk.Text(
            func_frame, width=60, state=tk.DISABLED,
            font=('Consolas', 8), wrap=tk.NONE,
            yscrollcommand=func_vsb.set,
            xscrollcommand=func_hsb.set,
        )
        func_vsb.config(command=self._func_list.yview)
        func_hsb.config(command=self._func_list.xview)
        func_hsb.pack(side=tk.BOTTOM, fill=tk.X)
        func_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._func_list.pack(fill=tk.BOTH, expand=True)
        self._func_list.bind('<Motion>',   self._on_funclist_motion)
        self._func_list.bind('<Leave>',    self._on_funclist_leave)
        self._func_list.bind('<Button-1>', self._on_funclist_click)

        canvas = tk.Canvas(grid_frame, bg='white')
        vsb = tk.Scrollbar(grid_frame, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner = tk.Frame(canvas, bg='white')
        self._inner_id = canvas.create_window((0, 0), window=self._inner, anchor='nw')
        self._canvas = canvas
        self._inner.bind('<Configure>',
                         lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', self._on_canvas_resize)
        # mouse-wheel scrolling
        canvas.bind_all('<MouseWheel>',
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), 'units'))

        self._cells = {}
        self._last_cell_w = 120
        for n in range(1, N_MAX + 1):
            row, col = divmod(n - 1, COLS)
            frame = tk.Frame(self._inner, bg=COLOR_UNKNOWN, relief=tk.RAISED,
                             bd=2, width=120, height=44, cursor='hand2')
            frame.grid(row=row, column=col, padx=1, pady=1, sticky='nsew')
            frame.grid_propagate(False)

            num_lbl = tk.Label(frame, text=str(n), bg=COLOR_UNKNOWN,
                               font=('Arial', 9, 'bold'), anchor='center')
            num_lbl.place(relx=0.5, rely=0.28, anchor='center')

            word_lbl = tk.Label(frame, text='', bg=COLOR_UNKNOWN,
                                font=('Arial', 7), fg='#2a5a2a',
                                wraplength=114, justify='center', anchor='center')
            word_lbl.place(relx=0.5, rely=0.72, anchor='center')

            q_lbl = tk.Label(frame, text='?', bg='#7788aa', fg='white',
                            font=('Arial', 6, 'bold'), cursor='question_arrow',
                            padx=1, pady=0)
            q_lbl.place(relx=1.0, rely=0.0, anchor='ne', x=-1, y=1)
            q_lbl.bind('<Enter>',    lambda e, num=n: self._on_q_enter(num))
            q_lbl.bind('<Leave>',    lambda e, num=n: self._on_q_leave(num))
            q_lbl.bind('<Button-1>', lambda e, num=n: self._on_q_click(num) or 'break')
            self._q_labels[n] = q_lbl

            for widget in (frame, num_lbl, word_lbl):
                widget.bind('<Button-1>', lambda e, num=n: self._on_click(num))

            self._cells[n] = (frame, num_lbl, word_lbl)

        # set initial 85/15 sash after window has been drawn
        self.root.after(100, self._set_initial_sash)

        # --- log output ---
        log_frame = tk.LabelFrame(self._learn_page, text="Ausgabe", padx=4, pady=4)
        log_frame.pack(fill=tk.X, padx=6, pady=(0, 4))
        self._log = scrolledtext.ScrolledText(log_frame, height=8, state=tk.DISABLED,
                                              font=('Consolas', 9), wrap=tk.WORD)
        self._log.pack(fill=tk.X)

        # redirect stdout → log widget
        sys.stdout = _LogRedirector(self._log, sys.__stdout__)

        # build the (initially hidden) grammar-test page
        self._build_test_page()

    def _on_oracle_toggle(self):
        global _oracle_style
        _oracle_style = 'hybrid2' if self._stat_oracle_var.get() else 'hybrid3'

    def _lock_oracle_toggle(self):
        self._oracle_cb.config(state=tk.DISABLED)
        _Tooltip(
            self._oracle_cb,
            "Einstellung gesperrt – das Orakel kann nach Beginn des Lernens "
            "nicht mehr geändert werden.",
        )

    def _set_initial_sash(self):
        w = self._main_pane.winfo_width()
        if w > 20:
            self._main_pane.sash_place(0, int(w * 0.75), 0)
        else:
            self.root.after(100, self._set_initial_sash)

    def _on_canvas_resize(self, event):
        self._canvas.itemconfig(self._inner_id, width=event.width)
        cell_w = max(30, (event.width - COLS * 4) // COLS)
        if cell_w != self._last_cell_w:
            self._last_cell_w = cell_w
            cell_h = max(26, int(cell_w * 0.37))
            self._resize_cells(cell_w, cell_h)

    def _resize_cells(self, cell_w, cell_h):
        font_num  = max(6, int(cell_w * 0.08))
        font_word = max(5, int(cell_w * 0.065))
        for n, (frame, num_lbl, word_lbl) in self._cells.items():
            frame.config(width=cell_w, height=cell_h)
            num_lbl.config(font=('Arial', font_num, 'bold'))
            word_lbl.config(font=('Arial', font_word), wraplength=max(10, cell_w - 6))

    def _refresh_func_list(self):
        lex = getattr(fmr, 'kenntnis_lexikon', {})
        lines = []
        sorted_keys = sorted(lex.keys())
        new_func_line_map = {}
        new_line_func_map = {}
        for i, key in enumerate(sorted_keys):
            entry = lex[key]
            try:
                lines.append(entry.present(printout=False))
            except Exception:
                lines.append(str(key))
            ln = i + 1
            new_func_line_map[key] = ln
            new_line_func_map[ln] = key
        self._func_line_map = new_func_line_map
        self._line_func_map = new_line_func_map
        text = '\n'.join(lines)
        self._func_list.configure(state=tk.NORMAL)
        self._func_list.delete('1.0', tk.END)
        self._func_list.insert(tk.END, text)
        self._func_list.configure(state=tk.DISABLED)
        # Re-apply function-list highlight (tags are cleared by delete)
        active = self._pinned_root or self._hover_root
        if active:
            self._apply_func_highlight(active)

    # ── periodic update ──────────────────────────────────────────────────────
    def _check_cycle(self):
        waiting = _waiting_event.is_set()
        if waiting:
            self._status_var.set("⏸ Algorithmus wartet — Zahl anklicken, um Wort einzugeben")
            self.root.configure(bg=COLOR_WAITING)
            self._learn_page.configure(bg=COLOR_WAITING)
        else:
            self._status_var.set("▶ Algorithmus läuft …")
            self.root.configure(bg='#f5f5f5')
            self._learn_page.configure(bg='#f5f5f5')

        # refresh cell states based on current lexicon
        lex = getattr(fmr, 'kenntnis_lexikon', {})
        lex_keys = frozenset(
            (key, lex[key].confirmed_maximum, sum(len(c) for c in lex[key].inputrange))
            for key in lex
        )
        if lex_keys != self._last_lex_keys:
            self._last_lex_keys = lex_keys
            self._recompute_owners(lex)
        for n in range(1, N_MAX + 1):
            if n in self._pending:
                # Keep orange until the algorithm is waiting for the next input AND
                # the submitted word is confirmed in the lexicon.
                if not waiting:
                    continue   # algorithm still processing → stay orange
                conf_word = grammar_generate(n, lex, only_confirmed=True)
                if conf_word != self._pending_words.get(n):
                    continue   # submitted word not yet confirmed → stay orange
                # Algorithm is idle and word is confirmed → fall through
                self._pending.discard(n)
                self._pending_words.pop(n, None)
            conf_word = grammar_generate(n, lex, only_confirmed=True)
            if conf_word:
                new_word = conf_word
                user_confirmed = n in self._user_submitted
                was_confirmed = n in self._confirmed
                changed = not was_confirmed or self._words.get(n) != new_word
                self._pending.discard(n)
                self._abstract.discard(n)
                self._confirmed.add(n)
                if changed:
                    self._words[n] = new_word
                if user_confirmed:
                    # directly entered by operator → dark green, locked
                    if changed:
                        self._set_cell_style(n, COLOR_KNOWN, new_word, clickable=False)
                else:
                    # confirmed by algorithm (forced) → light green, still clickable
                    if changed or (was_confirmed and self._words.get(n) == new_word):
                        self._set_cell_style(n, COLOR_FORCED, new_word, clickable=True)
            else:
                abs_word = grammar_generate(n, lex)
                if abs_word:
                    # only reachable via abstraction → show word but keep clickable
                    new_word = abs_word
                    changed = n not in self._abstract or self._words.get(n) != new_word
                    self._pending.discard(n)
                    self._abstract.add(n)
                    if changed:
                        self._words[n] = new_word
                        self._set_cell_style(n, COLOR_ABSTRACT, new_word, clickable=True)
                else:
                    # nothing known yet — restore to unknown/pending appearance
                    # This also fires after UNDO when a previously confirmed cell
                    # is no longer covered by the restored lexicon.
                    if n in self._abstract or n in self._confirmed:
                        self._abstract.discard(n)
                        self._confirmed.discard(n)
                        self._user_submitted.discard(n)  # allow re-entry
                        self._words.pop(n, None)
                        color = COLOR_PENDING if n in self._pending else COLOR_UNKNOWN
                        self._set_cell_style(n, color, '', clickable=True)

        self.root.after(400, self._check_cycle)
        self._refresh_func_list()

    # ── user actions ─────────────────────────────────────────────────────────
    def _on_click(self, number):
        if not _waiting_event.is_set():
            messagebox.showinfo("Hinweis",
                                "Der Algorithmus benötigt gerade keine Eingabe.",
                                parent=self.root)
            return

        # dark-green (user-confirmed) cells are permanently locked
        if number in self._user_submitted and number in self._confirmed:
            return

        # Pre-fill with current word for abstract or forced-confirmed cells
        is_forced   = number in self._confirmed and number not in self._user_submitted
        is_abstract = number in self._abstract
        prefill = self._words.get(number, '') if (is_abstract or is_forced) else ''
        if is_forced:
            prompt = f"Zahlwort für {number} korrigieren (zwangsbestätigt):"
        elif is_abstract:
            prompt = f"Zahlwort für {number} bestätigen oder korrigieren:"
        else:
            prompt = f"Zahlwort für {number}:"
        wort = simpledialog.askstring(
            "Zahlwort eingeben",
            prompt,
            initialvalue=prefill,
            parent=self.root
        )
        if wort is None:  # user cancelled
            return
        wort = wort.strip().lower()
        if not wort:
            messagebox.showwarning("Hinweis", "Bitte ein Wort eingeben.", parent=self.root)
            return

        self._abstract.discard(number)
        self._confirmed.discard(number)   # allow re-learning if forced-confirmed
        self._user_submitted.add(number)
        self._pending_words[number] = wort
        self._set_cell_style(number, COLOR_PENDING, wort)
        self._pending.add(number)
        if not self._oracle_locked:
            self._oracle_locked = True
            self._lock_oracle_toggle()
            _start_event.set()  # unblock thread so it can create the Oracle
        _input_queue.put(f"{number} {wort}")

    def _send_stop(self):
        if messagebox.askyesno("STOP", "Lernen wirklich beenden?", parent=self.root):
            if _waiting_event.is_set():
                _input_queue.put("STOP")
            else:
                _input_queue.put("STOP")  # will be picked up on next wait

    def _send_undo(self):
        if _waiting_event.is_set():
            _input_queue.put("UNDO")

    # ── grammar-test page ─────────────────────────────────────────────────────
    def _build_test_page(self):
        page = self._test_page
        self._test_last_tree = None

        top = tk.Frame(page)
        top.pack(fill=tk.X, padx=6, pady=4)
        tk.Button(top, text="◀ LERNE WEITER", bg='#2e8b57', fg='white',
                  command=self._show_learn_page).pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="Grammatik testen", font=('Arial', 14, 'bold'),
                 fg='#1a4a8a').pack(side=tk.LEFT, padx=12)

        inp = tk.Frame(page)
        inp.pack(fill=tk.X, padx=10, pady=(10, 4))
        tk.Label(inp, text="Wort oder Zahl eingeben:",
                 font=('Arial', 11)).pack(side=tk.LEFT)
        self._test_entry = tk.Entry(inp, font=('Consolas', 13), width=30)
        self._test_entry.pack(side=tk.LEFT, padx=8)
        self._test_entry.bind('<Return>', self._on_test_submit)
        tk.Button(inp, text="Übersetzen",
                  command=self._on_test_submit).pack(side=tk.LEFT)

        self._test_result_var = tk.StringVar(value='')
        tk.Label(page, textvariable=self._test_result_var,
                 font=('Consolas', 16, 'bold'), fg='#2a5a2a',
                 anchor='w').pack(fill=tk.X, padx=12, pady=(2, 6))

        constr = tk.LabelFrame(page, text="Konstruktion", padx=6, pady=4)
        constr.pack(fill=tk.X, padx=10, pady=(0, 6))
        self._test_constr_var = tk.StringVar(value='')
        tk.Label(constr, textvariable=self._test_constr_var,
                 font=('Consolas', 12), fg='#553300',
                 anchor='w', justify='left', wraplength=900).pack(fill=tk.X)

        graph = tk.LabelFrame(page, text="Konstruktionsgraph", padx=4, pady=4)
        graph.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self._test_canvas = tk.Canvas(graph, bg='white', highlightthickness=0)
        self._test_canvas.pack(fill=tk.BOTH, expand=True)
        self._test_canvas.bind(
            '<Configure>',
            lambda e: self._test_last_tree and self._draw_tree(self._test_last_tree))

    def _show_test_page(self):
        self._learn_page.pack_forget()
        self._test_page.pack(fill=tk.BOTH, expand=True)
        self._test_entry.focus_set()

    def _show_learn_page(self):
        self._test_page.pack_forget()
        self._learn_page.pack(fill=tk.BOTH, expand=True)

    def _on_test_submit(self, event=None):
        raw = self._test_entry.get().strip()
        if not raw:
            return
        lex = getattr(fmr, 'kenntnis_lexikon', {})

        def _clear():
            self._test_constr_var.set('')
            self._test_last_tree = None
            self._test_canvas.delete('all')

        if not lex:
            self._test_result_var.set(">> (es wurden noch keine Wörter gelernt)")
            _clear()
            return

        is_number = raw.lstrip('-').isdigit()
        if is_number:
            number = int(raw)
            word = grammar_generate(number, lex)
            if not word:
                self._test_result_var.set(f">> {number} ist (noch) nicht erzeugbar")
                _clear()
                return
            self._test_result_var.set(f">> {word}")
        else:
            word = raw.lower()
            number = grammar_parse(word, lex)
            if number == -1:
                self._test_result_var.set(f">> '{word}' ist nicht erkennbar")
                _clear()
                return
            self._test_result_var.set(f">> {number}")

        info = explain_generate(number, lex)
        if info is None:
            self._test_constr_var.set('(keine Konstruktion verfügbar)')
            self._test_last_tree = None
            self._test_canvas.delete('all')
            return
        _, tree_repr, _ = info
        self._test_constr_var.set(tree_repr)
        self._draw_tree(self._parse_construction(tree_repr))

    @staticmethod
    def _parse_construction(s):
        """Parse a 'root(child,child)' construction string into a nested
        (label, [children]) tree."""
        pos = 0
        n = len(s)

        def parse_node():
            nonlocal pos
            start = pos
            while pos < n and s[pos] not in '(),':
                pos += 1
            label = s[start:pos].strip()
            children = []
            if pos < n and s[pos] == '(':
                pos += 1
                while True:
                    children.append(parse_node())
                    if pos < n and s[pos] == ',':
                        pos += 1
                        continue
                    if pos < n and s[pos] == ')':
                        pos += 1
                    break
            return (label, children)

        return parse_node()

    def _draw_tree(self, tree):
        self._test_last_tree = tree
        canvas = self._test_canvas
        canvas.delete('all')

        nodes = []   # [label, x, depth]
        edges = []   # (parent_idx, child_idx)
        leaf_x = [0]
        max_depth = [0]

        def place(node, depth):
            label, children = node
            max_depth[0] = max(max_depth[0], depth)
            if children:
                child_idxs = [place(c, depth + 1) for c in children]
                x = sum(nodes[ci][1] for ci in child_idxs) / len(child_idxs)
            else:
                x = leaf_x[0]
                leaf_x[0] += 1
                child_idxs = []
            idx = len(nodes)
            nodes.append([label, x, depth])
            for ci in child_idxs:
                edges.append((idx, ci))
            return idx

        place(tree, 0)
        if not nodes:
            return

        n_leaves = max(1, leaf_x[0])
        depth = max(1, max_depth[0])
        canvas.update_idletasks()
        W = max(canvas.winfo_width(), 200)
        H = max(canvas.winfo_height(), 150)
        margin_x, margin_y = 60, 40

        def px(x):
            if n_leaves == 1:
                return W / 2
            return margin_x + x * (W - 2 * margin_x) / (n_leaves - 1)

        def py(d):
            return margin_y + d * (H - 2 * margin_y) / depth

        for p, c in edges:
            canvas.create_line(px(nodes[p][1]), py(nodes[p][2]),
                               px(nodes[c][1]), py(nodes[c][2]),
                               fill='#888888', width=2)

        for label, x, d in nodes:
            cx, cy = px(x), py(d)
            is_func = '_' in label
            fill = '#cfe6ff' if is_func else '#d6f5d6'
            outline = '#2a7ab0' if is_func else '#2e8b57'
            w = max(44, 7 * len(label) + 14)
            canvas.create_rectangle(cx - w / 2, cy - 14, cx + w / 2, cy + 14,
                                    fill=fill, outline=outline, width=2)
            canvas.create_text(cx, cy, text=label, font=('Consolas', 9))

    # ── helpers ──────────────────────────────────────────────────────────────
    def _recompute_owners(self, lex):
        """Recompute the number→root mapping for every cell. Called when lex changes."""
        new_owner = {}
        for n in range(1, N_MAX + 1):
            _, root = grammar_generate(n, lex, only_confirmed=False, _return_source=True)
            if root:
                new_owner[n] = root
        self._owner = new_owner
        if self._pinned_root and self._pinned_root not in lex:
            self._pinned_root = None
        active = self._pinned_root or self._hover_root
        if active:
            self._set_highlight(active)

    def _apply_func_highlight(self, root):
        """Highlight the function-list line for root (tags survive disabled state)."""
        self._func_list.tag_remove('hl_func', '1.0', tk.END)
        line_no = self._func_line_map.get(root)
        if line_no is not None:
            self._func_list.tag_add('hl_func', f'{line_no}.0', f'{line_no}.end+1c')
            self._func_list.tag_config('hl_func', background='#ffe080')
            self._func_list.see(f'{line_no}.0')

    def _set_highlight(self, root):
        """Highlight function line and outline all cells generated by root."""
        self._apply_func_highlight(root)
        for n, (frame, num_lbl, word_lbl) in self._cells.items():
            if self._owner.get(n) == root:
                frame.configure(highlightbackground='#cc2222', highlightthickness=3)
            else:
                frame.configure(highlightthickness=0)

    def _clear_highlight(self):
        """Remove all highlights from function list and cells."""
        self._func_list.tag_remove('hl_func', '1.0', tk.END)
        for frame, num_lbl, word_lbl in self._cells.values():
            frame.configure(highlightthickness=0)

    def _on_q_enter(self, n):
        root = self._owner.get(n)
        if root:
            self._hover_root = root
            self._set_highlight(root)

    def _on_q_leave(self, n):
        self._hover_root = None
        if self._pinned_root:
            self._set_highlight(self._pinned_root)
        else:
            self._clear_highlight()

    def _on_q_click(self, n):
        root = self._owner.get(n)
        if not root:
            return
        if self._pinned_root == root:
            self._pinned_root = None
            self._clear_highlight()
        else:
            self._pinned_root = root
            self._set_highlight(root)

    def _on_funclist_motion(self, event):
        idx = self._func_list.index(f'@{event.x},{event.y}')
        line_no = int(idx.split('.')[0])
        root = self._line_func_map.get(line_no)
        if root != self._hover_root:
            self._hover_root = root
            if root:
                self._set_highlight(root)
            elif self._pinned_root:
                self._set_highlight(self._pinned_root)
            else:
                self._clear_highlight()

    def _on_funclist_leave(self, event):
        self._hover_root = None
        if self._pinned_root:
            self._set_highlight(self._pinned_root)
        else:
            self._clear_highlight()

    def _on_funclist_click(self, event):
        idx = self._func_list.index(f'@{event.x},{event.y}')
        line_no = int(idx.split('.')[0])
        root = self._line_func_map.get(line_no)
        if not root:
            return
        if self._pinned_root == root:
            self._pinned_root = None
            self._clear_highlight()
        else:
            self._pinned_root = root
            self._set_highlight(root)

    def _set_cell_style(self, number, color, word=None, clickable=True):
        cell = self._cells.get(number)
        if not cell:
            return
        frame, num_lbl, word_lbl = cell
        frame.configure(bg=color)
        num_lbl.configure(bg=color)
        word_lbl.configure(bg=color)
        if word is not None:
            word_lbl.configure(text=word)
        if not clickable:
            cursor = ''
            for widget in (frame, num_lbl, word_lbl):
                widget.unbind('<Button-1>')
                widget.configure(cursor=cursor) if hasattr(widget, 'configure') else None
        else:
            # Re-enable clicking in case it was previously disabled (e.g. after undo).
            for widget in (frame, num_lbl, word_lbl):
                widget.bind('<Button-1>', lambda e, num=number: self._on_click(num))
                widget.configure(cursor='hand2') if hasattr(widget, 'configure') else None


# ── stdout → log widget ──────────────────────────────────────────────────────
class _LogRedirector:
    def __init__(self, widget, fallback):
        self._widget  = widget
        self._fallback = fallback

    def write(self, text):
        try:
            self._widget.configure(state=tk.NORMAL)
            self._widget.insert(tk.END, text)
            self._widget.see(tk.END)
            self._widget.configure(state=tk.DISABLED)
        except Exception:
            self._fallback.write(text)

    def flush(self):
        pass


# ── background thread ────────────────────────────────────────────────────────
def _run_algorithm():
    # Signal that the GUI may accept input, then wait until the oracle
    # choice is locked (first word submitted) before creating the Oracle.
    _waiting_event.set()
    _start_event.wait()
    try:
        if _oracle_style == 'hybrid2':
            orakel = Oracle(
                'hybrid2',
                tolerance_factor=73.14,
                tolerance_factor_glatt=2**4,
                tolerance_factor_nichtglatt=2**9,
                search_engine=cached_openlibrary_search_count,
            )
        else:
            orakel = Oracle('hybrid3')
        learn_lexicon(range(1000), orakel, printb=True,
                      normalize=True, restricted_merge=False)
    except StopLearning:
        print("\n[Lernen beendet (STOP)]")
    except UndoLearning:
        pass  # should never propagate here; undo is handled inside learn_lexicon
    except Exception as exc:
        print(f"\n[Fehler im Algorithmus: {exc}]")
    finally:
        save_search_cache(search_cache, SEARCH_CACHE_PATH)
        _waiting_event.clear()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1100x750')
    root.configure(bg='#f5f5f5')

    gui = ZahlwortGUI(root)

    t = threading.Thread(target=_run_algorithm, daemon=True)
    t.start()

    root.mainloop()
