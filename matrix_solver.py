import tkinter as tk
from tkinter import ttk, messagebox
import sympy as sp

class MatrixSolver:
    """
    Handles all linear algebra operations independently of the GUI.
    Supports parsing from 2D grids, calculating formulas, and executing
    operations on dynamically evaluated matrix string expressions.
    """
    def __init__(self):
        self.matrix_a = None
        self.matrix_b = None
        self.matrix_a_size = (0, 0)
        self.matrix_b_size = (0, 0)
    
    def set_matrix_size(self, matrix_name, rows, cols):
        if matrix_name == 'A':
            self.matrix_a_size = (rows, cols)
            self.matrix_a = None
            return [[0 for _ in range(cols)] for _ in range(rows)]
        elif matrix_name == 'B':
            self.matrix_b_size = (rows, cols)
            self.matrix_b = None
            return [[0 for _ in range(cols)] for _ in range(rows)]
        else:
            raise ValueError("matrix_name must be 'A' or 'B'")

    def parse_matrix(self, matrix_input, expected_rows=None, expected_cols=None):
        try:
            if isinstance(matrix_input, list):
                data = []
                for row in matrix_input:
                    parsed_row = []
                    for val in row:
                        val_str = str(val).strip()
                        parsed_row.append(sp.sympify(val_str) if val_str else sp.Rational(0))
                    data.append(parsed_row)
            else:
                rows = [r.split(',') for r in matrix_input.split(';')]
                data = [[sp.sympify(val.strip()) if val.strip() else sp.Rational(0) for val in r] for r in rows]
            
            matrix = sp.Matrix(data)
            if expected_rows is not None and expected_cols is not None:
                if matrix.rows != expected_rows or matrix.cols != expected_cols:
                    raise ValueError(f"Expected {expected_rows}x{expected_cols}, got {matrix.shape}")
            return matrix
        except Exception as e:
            raise ValueError(f"Invalid Matrix entry: {e}")

    def update_matrix_a(self, grid_data):
        self.matrix_a = self.parse_matrix(grid_data, self.matrix_a_size[0], self.matrix_a_size[1])
        return self.matrix_a

    def update_matrix_b(self, grid_data):
        self.matrix_b = self.parse_matrix(grid_data, self.matrix_b_size[0], self.matrix_b_size[1])
        return self.matrix_b

    def evaluate_matrix_expression(self, expr_str, current_grid_a, current_grid_b):
        """
        Dynamically evaluates an expression string (like 'A*B + A**2') using 
        the matrices currently compiled from live user entries grid blocks.
        """
        expr_str = expr_str.strip()
        mat_a = self.update_matrix_a(current_grid_a)
        mat_b = self.update_matrix_b(current_grid_b)

        if not expr_str:
            return mat_a, "Matrix A (Default)"

        try:
            sanitized = expr_str.replace('^', '**')
            sanitized = sanitized.replace('A2', 'A**2').replace('B2', 'B**2')
            local_dict = {'A': mat_a, 'B': mat_b}
            result = sp.sympify(sanitized, locals=local_dict)
            return result, expr_str
        except Exception as e:
            raise ValueError(f"Failed to parse matrix mathematical expression '{expr_str}': {e}")

    @staticmethod
    def format_output(matrix, precision=4):
        if not isinstance(matrix, sp.Matrix):
            try: return round(float(matrix), precision)
            except: return str(matrix)
        return matrix.evalf(n=precision + 2).applyfunc(
            lambda x: round(float(x), precision) if x.is_Number else x
        )

    def transpose(self, matrix):
        if matrix is None: raise ValueError("Matrix is empty.")
        return matrix.T

    def determinant(self, matrix):
        if matrix is None: raise ValueError("Matrix is empty.")
        if not matrix.is_square: raise ValueError("Determinant requires a square matrix.")
        return matrix.det()

    def inverse(self, matrix):
        if matrix is None: raise ValueError("Matrix is empty.")
        if not matrix.is_square: raise ValueError("Inverse requires a square matrix.")
        if matrix.det() == 0: raise ValueError("Singular matrix (determinant is zero).")
        return matrix.inv()

    def adjoint(self, matrix):
        if matrix is None: raise ValueError("Matrix is empty.")
        if not matrix.is_square: raise ValueError("Adjoint requires a square matrix.")
        return matrix.adjugate()

    def power(self, matrix, p):
        if matrix is None: raise ValueError("Matrix is empty.")
        if not matrix.is_square: raise ValueError("Power requires a square matrix.")
        return matrix**p

    def add(self, matrix_a, matrix_b):
        if matrix_a is None or matrix_b is None: raise ValueError("Both matrices must be populated.")
        if matrix_a.shape != matrix_b.shape: raise ValueError("Dimension mismatch for addition.")
        return matrix_a + matrix_b

    def multiply(self, matrix_a, matrix_b):
        if matrix_a is None or matrix_b is None: raise ValueError("Both matrices must be populated.")
        if matrix_a.cols != matrix_b.rows: raise ValueError("Incompatible dimensions for multiplication.")
        return matrix_a * matrix_b


class MatrixGrid:
    def __init__(self, parent, title, solver, matrix_name):
        self.parent = parent
        self.title = title
        self.solver = solver
        self.matrix_name = matrix_name
        self.grid_frame = None
        self.entries = []
        self.size = (0, 0)
        self.create_widgets()
    
    def create_widgets(self):
        ttk.Label(self.parent, text=self.title, font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(5, 2))
        control_frame = ttk.Frame(self.parent, style="Card.TFrame")
        control_frame.pack(fill='x', pady=2)
        
        size_btn = ttk.Button(control_frame, text="Set Dimensions", style="Action.TButton", command=self.open_size_dialog)
        size_btn.pack(side="left", padx=2)
        
        self.grid_frame = ttk.Frame(self.parent, style="Card.TFrame")
        self.grid_frame.pack(pady=5, fill='both', expand=True)
    
    def open_size_dialog(self):
        dialog = tk.Toplevel(self.parent)
        dialog.title(f"Set {self.matrix_name} Size")
        dialog.geometry("250x180")
        dialog.resizable(False, False)
        dialog.configure(bg="#1e293b")
        dialog.transient(self.parent)
        dialog.grab_set()
        
        tk.Label(dialog, text="Rows:", bg="#1e293b", fg="#ffffff", font=("Segoe UI", 9)).pack(pady=(10,2))
        rows_var = tk.StringVar(value="2")
        rows_spin = ttk.Spinbox(dialog, from_=1, to=10, textvariable=rows_var, width=8, justify='center')
        rows_spin.pack()
        
        tk.Label(dialog, text="Columns:", bg="#1e293b", fg="#ffffff", font=("Segoe UI", 9)).pack(pady=(10,2))
        cols_var = tk.StringVar(value="2")
        cols_spin = ttk.Spinbox(dialog, from_=1, to=10, textvariable=cols_var, width=8, justify='center')
        cols_spin.pack()
        
        def set_size():
            try:
                rows = int(rows_var.get())
                cols = int(cols_var.get())
                empty_grid = self.solver.set_matrix_size(self.matrix_name, rows, cols)
                self.create_grid(rows, cols, empty_grid)
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integers!")
        ttk.Button(dialog, text="Apply", command=set_size).pack(pady=15)
    
    def create_grid(self, rows, cols, initial_data=None):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.entries = []
        self.size = (rows, cols)
        
        for i in range(rows):
            row_entries = []
            row_frame = ttk.Frame(self.grid_frame, style="Card.TFrame")
            row_frame.pack(fill='x', pady=2)
            for j in range(cols):
                entry = tk.Entry(row_frame, width=6, justify='center', font=('Consolas', 11),
                                 bg="#0f172a", fg="#f8fafc", insertbackground="white",
                                 relief="flat", highlightthickness=1, highlightbackground="#334155")
                entry.pack(side='left', padx=3, ipady=2)
                
                val_to_insert = "0"
                if initial_data and i < len(initial_data) and j < len(initial_data[i]):
                    val_to_insert = str(initial_data[i][j])
                entry.insert(0, val_to_insert)
                row_entries.append(entry)
            self.entries.append(row_entries)
    
    def get_grid_data(self):
        return [[entry.get() for entry in row_entries] for row_entries in self.entries]