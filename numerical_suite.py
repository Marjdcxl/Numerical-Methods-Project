import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp

# --- CRITICAL BLENDING IMPORT CONNECTION ---
from matrix_solver import MatrixSolver, MatrixGrid

class RootFinder:
    @staticmethod
    def parse_function(func_str):
        x = sp.symbols('x')
        try:
            expr = sp.simplify(func_str.replace('^', '**'))
            # Added math module to lambdify fallback for single float execution safety
            f = sp.lambdify(x, expr, modules=['numpy', 'math'])
            df = sp.lambdify(x, sp.diff(expr, x), modules=['numpy', 'math'])
            return f, df, expr
        except Exception as e:
            raise ValueError(f"Invalid function expression: {e}")

    @staticmethod
    def incremental_search(f, start, end, step=0.01):
        iterations = []
        x_prev = start
        idx = 1
        while x_prev < end:
            x_next = x_prev + step
            if x_next > end:
                x_next = end
            try:
                f_prev, f_next = float(f(x_prev)), float(f(x_next))
                prod = f_prev * f_next
                prod_str = "<0" if prod < 0 else (">0" if prod > 0 else "0")
                remark = "Root Found" if prod <= 0 else "Proceed"
                
                iterations.append([
                    idx, f"{x_prev:.4f}", f"{x_next:.4f}", 
                    f"{f_prev:.4f}", f"{f_next:.4f}", prod_str, remark
                ])
                if prod <= 0: 
                    return x_next, iterations
            except Exception as e: 
                print(f"Evaluation error: {e}")
                break
            x_prev = x_next
            idx += 1
        return None, iterations

    @staticmethod
    def bisection(f, a, b, tol=1e-5, max_iter=50):
        iterations = []
        if float(f(a)) * float(f(b)) >= 0: 
            raise ValueError("f(a) and f(b) must have opposite signs for Bisection.")
        for i in range(max_iter):
            c = (a + b) / 2
            fc = float(f(c))
            iterations.append([i+1, f"{a:.4f}", f"{b:.4f}", f"{c:.6f}", f"{fc:.6f}"])
            if abs(fc) < tol or (b - a) / 2 < tol: return c, iterations
            if float(f(a)) * fc < 0: b = c
            else: a = c
        return c, iterations

    @staticmethod
    def regula_falsi(f, a, b, tol=1e-5, max_iter=50):
        iterations = []
        if float(f(a)) * float(f(b)) >= 0: 
            raise ValueError("f(a) and f(b) must have opposite signs.")
        for i in range(max_iter):
            fa, fb = float(f(a)), float(f(b))
            c = (a * fb - b * fa) / (fb - fa)
            fc = float(f(c))
            iterations.append([i+1, f"{a:.4f}", f"{b:.4f}", f"{c:.6f}", f"{fc:.6f}"])
            if abs(fc) < tol: return c, iterations
            if fa * fc < 0: b = c
            else: a = c
        return c, iterations

    @staticmethod
    def newton_raphson(f, df, x0, tol=1e-5, max_iter=50):
        iterations = []
        x_curr = x0
        for i in range(max_iter):
            fx = float(f(x_curr))
            dfx = float(df(x_curr))
            if abs(dfx) < 1e-12: raise ValueError("Derivative near zero. Method fails.")
            x_next = x_curr - fx / dfx
            iterations.append([i+1, f"{x_curr:.6f}", f"{fx:.6f}", f"{x_next:.6f}"])
            if abs(x_next - x_curr) < tol: return x_next, iterations
            x_curr = x_next
        return x_curr, iterations

    @staticmethod
    def secant(f, x0, x1, tol=1e-5, max_iter=50):
        iterations = []
        for i in range(max_iter):
            f0, f1 = float(f(x0)), float(f(x1))
            if abs(f1 - f0) < 1e-12: raise ValueError("Denominator near zero.")
            x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
            iterations.append([i+1, f"{x0:.4f}", f"{x1:.4f}", f"{x_next:.6f}"])
            if abs(x_next - x1) < tol: return x_next, iterations
            x0, x1 = x1, x_next
        return x1, iterations


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculus & Matrix Pro - Numerical Suite")
        self.geometry("1250x950")
        
        self.solver = MatrixSolver()
        self.colors = {
            "bg": "#0f172a", "card": "#1e293b", "accent": "#38bdf8",
            "success": "#4ade80", "warning": "#fbbf24", "text": "#f8fafc",
            "text_dim": "#94a3b8", "border": "#334155"
        }

        self.configure(bg=self.colors["bg"])
        self.setup_styles()

        header = tk.Frame(self, bg=self.colors["bg"], pady=15)
        header.pack(fill="x")
        tk.Label(header, text="NUMERICAL METHODS SUITE", font=("Segoe UI", 24, "bold"), 
                 bg=self.colors["bg"], fg=self.colors["accent"]).pack()
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=20, pady=(0, 20))

        self.init_root_tab()
        self.init_matrix_tab()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=self.colors["card"], foreground=self.colors["text_dim"], 
                        padding=[20, 10], font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", self.colors["accent"])], foreground=[("selected", "#ffffff")])
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Card.TFrame", background=self.colors["card"], relief="flat")
        style.configure("TLabel", background=self.colors["card"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("Action.TButton", font=("Segoe UI", 9, "bold"), background=self.colors["accent"], foreground="#ffffff", borderwidth=0, padding=[10, 5])
        style.map("Action.TButton", background=[("active", "#0ea5e9")])
        style.configure("Treeview", background=self.colors["card"], foreground=self.colors["text"], 
                        fieldbackground=self.colors["card"], rowheight=30, borderwidth=0, font=("Consolas", 9))
        style.configure("Treeview.Heading", background=self.colors["border"], foreground=self.colors["accent"], 
                        font=("Segoe UI", 9, "bold"), relief="flat")

    def init_root_tab(self):
        root_frame = ttk.Frame(self.notebook)
        self.notebook.add(root_frame, text="   ROOT FINDING   ")

        side_panel = ttk.Frame(root_frame, style="Card.TFrame", padding=20)
        side_panel.pack(side="left", fill="y", padx=(0, 2))

        self.create_label_entry(side_panel, "Function f(x):", "x**2 - 4", "func")
        
        ttk.Label(side_panel, text="Method:").pack(anchor="w", pady=(10, 0))
        self.method_var = tk.StringVar(value="Bisection")
        self.method_menu = ttk.Combobox(side_panel, textvariable=self.method_var, values=["Incremental", "Bisection", "Regula Falsi", "Newton Raphson", "Secant"], state="readonly")
        self.method_menu.pack(fill="x", pady=5)
        self.method_menu.bind("<<ComboboxSelected>>", self.on_method_change)

        # Labels references kept explicitly to mutate values dynamically via combo selector
        self.lbl_p1 = ttk.Label(side_panel, text="Initial Guess / Start:")
        self.lbl_p1.pack(anchor="w", pady=(10, 0))
        self.p1_entry = tk.Entry(side_panel, bg=self.colors["bg"], fg=self.colors["text"], insertbackground="white", relief="flat")
        self.p1_entry.insert(0, "0")
        self.p1_entry.pack(fill="x", pady=5, ipady=3)

        self.lbl_p2 = ttk.Label(side_panel, text="Second Guess / End:")
        self.lbl_p2.pack(anchor="w", pady=(10, 0))
        self.p2_entry = tk.Entry(side_panel, bg=self.colors["bg"], fg=self.colors["text"], insertbackground="white", relief="flat")
        self.p2_entry.insert(0, "3")
        self.p2_entry.pack(fill="x", pady=5, ipady=3)

        self.lbl_p3 = ttk.Label(side_panel, text="Stopping Tolerance:")
        self.lbl_p3.pack(anchor="w", pady=(10, 0))
        self.p3_entry = tk.Entry(side_panel, bg=self.colors["bg"], fg=self.colors["text"], insertbackground="white", relief="flat")
        self.p3_entry.insert(0, "0.00001")
        self.p3_entry.pack(fill="x", pady=5, ipady=3)

        ttk.Button(side_panel, text="RUN ANALYSIS", style="Action.TButton", command=self.solve_root).pack(fill="x", pady=30)

        main_display = ttk.Frame(root_frame, style="TFrame")
        main_display.pack(side="right", expand=True, fill="both", padx=10)

        graph_card = ttk.Frame(main_display, style="Card.TFrame", padding=10)
        graph_card.pack(fill="both", expand=True, pady=(0, 10))

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor(self.colors["card"])
        self.ax.set_facecolor(self.colors["card"])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        table_card = ttk.Frame(main_display, style="Card.TFrame", padding=1)
        table_card.pack(fill="both", expand=True)

        x_scroll = ttk.Scrollbar(table_card, orient="horizontal")
        y_scroll = ttk.Scrollbar(table_card, orient="vertical")

        self.tree = ttk.Treeview(table_card, columns=("c1", "c2", "c3", "c4", "c5", "c6", "c7"), 
                                 show='headings', height=6, xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        x_scroll.config(command=self.tree.xview)
        y_scroll.config(command=self.tree.yview)
        
        x_scroll.pack(side="bottom", fill="x")
        y_scroll.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        self.lbl_root_result = ttk.Label(table_card, text="Root Found: None", font=("Segoe UI", 11, "bold"), foreground=self.colors["success"], padding=8)
        self.lbl_root_result.pack(side="bottom", fill="x", padx=5)

        self.on_method_change(None)

    def on_method_change(self, event):
        method = self.method_var.get()
        # Restore full visibility layout context parameters 
        self.lbl_p1.pack(anchor="w", pady=(10, 0))
        self.p1_entry.pack(fill="x", pady=5, ipady=3)
        self.lbl_p2.pack(anchor="w", pady=(10, 0))
        self.p2_entry.pack(fill="x", pady=5, ipady=3)
        self.lbl_p3.pack(anchor="w", pady=(10, 0))
        self.p3_entry.pack(fill="x", pady=5, ipady=3)

        if method == "Incremental":
            self.lbl_p1.config(text="Start Range (Lower x):")
            self.lbl_p2.config(text="End Range (Upper x):")
            self.lbl_p3.config(text="Step Size Delta:")
            self.p3_entry.delete(0, tk.END)
            self.p3_entry.insert(0, "0.05")
        elif method in ["Bisection", "Regula Falsi"]:
            self.lbl_p1.config(text="Lower Boundary (xl):")
            self.lbl_p2.config(text="Upper Boundary (xu):")
            self.lbl_p3.config(text="Stopping Tolerance:")
        elif method == "Newton Raphson":
            self.lbl_p1.config(text="Initial Guess (x0):")
            self.lbl_p2.pack_forget()
            self.p2_entry.pack_forget()
            self.lbl_p3.config(text="Stopping Tolerance:")
        elif method == "Secant":
            self.lbl_p1.config(text="Initial Guess (xi-1):")
            self.lbl_p2.config(text="Second Guess (xi):")
            self.lbl_p3.config(text="Stopping Tolerance:")

    def create_label_entry(self, parent, label_text, default, attr_name):
        ttk.Label(parent, text=label_text).pack(anchor="w", pady=(10, 0))
        entry = tk.Entry(parent, bg=self.colors["bg"], fg=self.colors["text"], insertbackground="white", relief="flat")
        entry.insert(0, default)
        entry.pack(fill="x", pady=5, ipady=3)
        setattr(self, f"{attr_name}_entry", entry)

    def solve_root(self):
        try:
            f, df, expr = RootFinder.parse_function(self.func_entry.get())
            method = self.method_var.get()
            p1 = float(self.p1_entry.get())
            
            try: p2 = float(self.p2_entry.get())
            except ValueError: p2 = 0.0

            try: p3 = float(self.p3_entry.get())
            except ValueError: p3 = 0.00001

            res, iters = None, []
            
            if method == "Incremental":
                cols = ("Idx", "x_prev", "x_next", "f(x_prev)", f"f(x_next)", "f(x_prev)*f(x_next)", "Remark")
                res, iters = RootFinder.incremental_search(f, p1, p2, step=p3)
            elif method == "Bisection":
                cols = ("Iter", "xl", "xu", "xr", "f(xr)")
                res, iters = RootFinder.bisection(f, p1, p2, tol=p3)
            elif method == "Regula Falsi":
                cols = ("Iter", "xl", "xu", "xr", "f(xr)")
                res, iters = RootFinder.regula_falsi(f, p1, p2, tol=p3)
            elif method == "Newton Raphson":
                cols = ("Iter", "X_curr", "f(X)", "X_next")
                res, iters = RootFinder.newton_raphson(f, df, p1, tol=p3)
            elif method == "Secant":
                cols = ("Iter", "X0", "X1", "X_new")
                res, iters = RootFinder.secant(f, p1, p2, tol=p3)

            # Rebuild dynamic columns sizes inside tree configuration maps
            self.tree.config(columns=[f"c{i+1}" for i in range(len(cols))])
            for i, col_name in enumerate(cols):
                self.tree.heading(f"c{i+1}", text=col_name)
                self.tree.column(f"c{i+1}", anchor="center", width=130, minwidth=100)

            for i in self.tree.get_children(): self.tree.delete(i)
            for row in iters:
                self.tree.insert("", "end", values=list(row))

            if res is not None:
                self.lbl_root_result.config(text=f"Root Found via {method}: x = {res:.6f}", foreground=self.colors["success"])
            else:
                self.lbl_root_result.config(text=f"Root Found via {method}: No converging root located in this range.", foreground=self.colors["warning"])

            self.ax.clear()
            
            if method in ["Incremental", "Bisection", "Regula Falsi", "Secant"]:
                x_vals = np.linspace(min(p1, p2) - 2, max(p1, p2) + 2, 400)
            else:
                x_vals = np.linspace(p1 - 4, p1 + 4, 400)
                
            v_func = np.vectorize(lambda val: float(f(val)))
            y_vals = v_func(x_vals)
            
            self.ax.plot(x_vals, y_vals, color=self.colors["accent"], label="f(x)", linewidth=2)
            self.ax.axhline(0, color=self.colors["text_dim"], linestyle='--', linewidth=1.2, alpha=0.7)
            self.ax.axvline(0, color=self.colors["text_dim"], linestyle='--', linewidth=1.2, alpha=0.7)
            self.ax.grid(True, linestyle=':', alpha=0.3, color=self.colors["text_dim"])
            
            if res is not None: 
                self.ax.plot(res, float(f(res)), 'o', color=self.colors["success"], markersize=8, label=f'Root: {res:.4f}')
            
            # Prevent astronomical layout compression on curve bounds
            y_min, y_max = np.min(y_vals), np.max(y_vals)
            if abs(y_max - y_min) > 100:
                self.ax.set_ylim(-20, 20)

            self.ax.legend()
            self.canvas.draw()
        except Exception as e: messagebox.showerror("Error", str(e))

    def init_matrix_tab(self):
        matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(matrix_frame, text="   MATRIX CALCULUS   ")

        top_ctrl = ttk.Frame(matrix_frame, style="TFrame", padding=10)
        top_ctrl.pack(fill="x")

        card_a = ttk.Frame(top_ctrl, style="Card.TFrame", padding=10)
        card_a.pack(side="left", fill="both", expand=True, padx=5)
        self.matrix_a_grid = MatrixGrid(card_a, "Matrix A Entries", self.solver, 'A')
        self.matrix_a_grid.create_grid(2, 2, [[1, 2], [3, 4]])

        card_b = ttk.Frame(top_ctrl, style="Card.TFrame", padding=10)
        card_b.pack(side="left", fill="both", expand=True, padx=5)
        self.matrix_b_grid = MatrixGrid(card_b, "Matrix B Entries", self.solver, 'B')
        self.matrix_b_grid.create_grid(2, 2, [[5, 6], [7, 8]])

        eq_input_frame = ttk.Frame(matrix_frame, style="Card.TFrame", padding=10)
        eq_input_frame.pack(fill="x", padx=15, pady=5)
        ttk.Label(eq_input_frame, text="Matrix Expression Input Field (Type mathematical formulation using 'A' and 'B', e.g., A*B + A2):", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.eq_entry = tk.Entry(eq_input_frame, bg=self.colors["bg"], fg=self.colors["text"], insertbackground="white", relief="flat")
        self.eq_entry.insert(0, "A")
        self.eq_entry.pack(fill="x", pady=5, ipady=3)

        op_frame = ttk.Frame(matrix_frame, style="TFrame", padding=10)
        op_frame.pack(fill="x")

        row1 = ttk.Frame(op_frame, style="TFrame")
        row1.pack(fill="x", pady=2)
        for t, c in [("Transpose", self.op_transpose), ("Determinant", self.op_det), 
                     ("Inverse", self.op_inv), ("Adjoint", self.op_adj), ("Power (^n)", self.op_power)]:
            ttk.Button(row1, text=t, command=c, style="Action.TButton").pack(side="left", padx=2, expand=True, fill="x")

        row2 = ttk.Frame(op_frame, style="TFrame")
        row2.pack(fill="x", pady=2)
        for t, c in [("Evaluate Matrix Expression", self.op_eval_only), ("A + B (Fallback Addition)", self.op_add), 
                     ("A * B (Fallback Multiplication)", self.op_mul)]:
            ttk.Button(row2, text=t, command=c, style="Action.TButton").pack(side="left", padx=2, expand=True, fill="x")

        res_card = ttk.Frame(matrix_frame, style="Card.TFrame", padding=15)
        res_card.pack(fill="both", expand=True, padx=15, pady=5)
        ttk.Label(res_card, text="COMPUTED PERFORMANCE / RESULT LOG", font=("Segoe UI", 10, "bold"), foreground=self.colors["success"]).pack(anchor="w")
        
        self.matrix_res_box = tk.Text(res_card, bg=self.colors["bg"], fg=self.colors["success"], font=("Consolas", 13), relief="flat", padx=10, pady=10)
        self.matrix_res_box.pack(fill="both", expand=True, pady=5)

    def get_expression_matrix(self):
        expr = self.eq_entry.get().strip()
        grid_a = self.matrix_a_grid.get_grid_data()
        grid_b = self.matrix_b_grid.get_grid_data()
        return self.solver.evaluate_matrix_expression(expr, grid_a, grid_b)

    def clean_number_string(self, num_str):
        try:
            val = float(num_str.strip())
            if val.is_integer():
                return str(int(val))
            formatted = f"{val:.4f}"
            return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
        except ValueError:
            return num_str.strip()

    def show_m_res(self, res, label="Result"):
        self.matrix_res_box.delete("1.0", tk.END)
        self.matrix_res_box.insert(tk.END, f"--- {label.upper()} ---\n\n")
        
        formatted = self.solver.format_output(res)
        if isinstance(res, sp.Matrix) or hasattr(res, 'rows'):
            lines = str(formatted).replace("Matrix(", "").rstrip(")").split("],")
            clean_str = ""
            for line in lines:
                row_content = line.replace("[", "").replace("]", "").strip()
                if row_content:
                    cleaned_items = [self.clean_number_string(item) for item in row_content.split(',')]
                    clean_row = ", ".join(cleaned_items)
                    clean_str += f"[  {clean_row}  ]\n"
            self.matrix_res_box.insert(tk.END, clean_str)
        else:
            self.matrix_res_box.insert(tk.END, self.clean_number_string(str(formatted)))

    def op_eval_only(self):
        try:
            matrix_res, actual_expr = self.get_expression_matrix()
            self.show_m_res(matrix_res, f"Evaluated Matrix Expression ({actual_expr})")
        except Exception as e: messagebox.showerror("Expression Error", str(e))

    def op_transpose(self):
        try:
            target_matrix, actual_expr = self.get_expression_matrix()
            self.show_m_res(self.solver.transpose(target_matrix), f"Transpose of: ({actual_expr})")
        except Exception as e: messagebox.showerror("Error", str(e))

    def op_det(self):
        try:
            target_matrix, actual_expr = self.get_expression_matrix()
            self.show_m_res(self.solver.determinant(target_matrix), f"Determinant of: ({actual_expr})")
        except Exception as e: messagebox.showerror("Error", str(e))

    def op_inv(self):
        try:
            target_matrix, actual_expr = self.get_expression_matrix()
            self.show_m_res(self.solver.inverse(target_matrix), f"Inverse of: ({actual_expr})")
        except Exception as e: messagebox.showerror("Error", str(e))

    def op_adj(self):
        try:
            target_matrix, actual_expr = self.get_expression_matrix()
            self.show_m_res(self.solver.adjoint(target_matrix), f"Adjoint of: ({actual_expr})")
        except Exception as e: messagebox.showerror("Error", str(e))

    def op_power(self):
        def apply_power():
            try:
                p = int(power_entry.get())
                target_matrix, actual_expr = self.get_expression_matrix()
                self.show_m_res(self.solver.power(target_matrix, p), f"Power ({actual_expr})^{p}")
                win.destroy()
            except Exception as e: messagebox.showerror("Error", str(e))
        win = tk.Toplevel(self)
        win.title("Exponent")
        win.configure(bg=self.colors["card"])
        tk.Label(win, text="Enter power n:", bg=self.colors["card"], fg="white").pack(padx=20, pady=5)
        power_entry = tk.Entry(win)
        power_entry.pack(padx=20, pady=5)
        ttk.Button(win, text="Apply", command=apply_power).pack(pady=10)

    def op_add(self):
        try:
            expr = self.eq_entry.get().strip()
            if expr and expr != "A":
                matrix_res, actual_expr = self.get_expression_matrix()
                self.show_m_res(matrix_res, f"Evaluated Expression: {actual_expr}")
            else:
                a = self.solver.update_matrix_a(self.matrix_a_grid.get_grid_data())
                b = self.solver.update_matrix_b(self.matrix_b_grid.get_grid_data())
                self.show_m_res(self.solver.add(a, b), "Default Matrix Addition (A + B)")
        except Exception as e: messagebox.showerror("Error", str(e))

    def op_mul(self):
        try:
            expr = self.eq_entry.get().strip()
            if expr and expr != "A":
                matrix_res, actual_expr = self.get_expression_matrix()
                self.show_m_res(matrix_res, f"Evaluated Expression: {actual_expr}")
            else:
                a = self.solver.update_matrix_a(self.matrix_a_grid.get_grid_data())
                b = self.solver.update_matrix_b(self.matrix_b_grid.get_grid_data())
                self.show_m_res(self.solver.multiply(a, b), "Default Matrix Multiplication (A * B)")
        except Exception as e: messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = App()
    app.mainloop()