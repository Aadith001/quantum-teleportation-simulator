import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

# -------------------------------
# Quantum functions
# -------------------------------
def normalize(state):
    norm = np.linalg.norm(state)
    return state / norm if norm != 0 else state

def kron(*matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

# Basic states and gates
zero = np.array([[1], [0]], dtype=complex)
one = np.array([[0], [1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
I = np.eye(2)
CNOT = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
], dtype=complex)

def measure_two_qubits(state):
    state = state.flatten()
    probs = np.abs(state)**2
    outcomes = [(0,0),(0,1),(1,0),(1,1)]
    outcome_probs = []
    for a,b in outcomes:
        mask = [i for i in range(8) if [(i>>2)&1, (i>>1)&1]==[a,b]]
        outcome_probs.append(probs[mask].sum())
    outcome_probs = np.array(outcome_probs)/np.sum(outcome_probs)
    idx = np.random.choice(4, p=outcome_probs)
    outcome = outcomes[idx]
    mask = [i for i in range(8) if [(i>>2)&1, (i>>1)&1]==list(outcome)]
    new_state = np.zeros_like(state)
    new_state[mask] = state[mask]
    new_state = normalize(new_state)
    return outcome, new_state

# -------------------------------
# GUI setup
# -------------------------------
root = tk.Tk()
root.title("Quantum Teleportation Simulator")
frame = ttk.Frame(root, padding=10)
frame.grid()

# Labels and inputs
ttk.Label(frame, text="Enter α (alpha):").grid(column=0, row=0, sticky="w")
alpha_entry = ttk.Entry(frame, width=10)
alpha_entry.insert(0, "0.6")
alpha_entry.grid(column=1, row=0)

ttk.Label(frame, text="Enter β (beta):").grid(column=2, row=0, sticky="w")
beta_entry = ttk.Entry(frame, width=10)
beta_entry.insert(0, "0.8")
beta_entry.grid(column=3, row=0)

# Output box
output_text = tk.Text(frame, height=6, width=70)
output_text.grid(column=0, row=3, columnspan=4, pady=10)

# Matplotlib figure
fig, ax = plt.subplots(figsize=(5,3))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(column=0, row=2, columnspan=4)

# Globals
state = None
psi = None
alice_result = None

# -------------------------------
# Plotting
# -------------------------------
def plot_state(state, title="Quantum State"):
    ax.clear()
    state = np.array(state).flatten()
    probs = np.abs(state)**2
    num_qubits = int(np.log2(len(probs)))
    labels = [f"|{i:0{num_qubits}b}>" for i in range(len(probs))]
    ax.bar(labels, probs, color='skyblue')
    ax.set_ylim(0,1)
    ax.set_title(title)
    canvas.draw()

# -------------------------------
# Simulation steps
# -------------------------------
def initialize():
    global psi, state
    try:
        alpha = complex(alpha_entry.get())
        beta = complex(beta_entry.get())
    except ValueError:
        messagebox.showerror("Invalid Input", "Enter valid numeric or complex values for α and β.")
        return

    psi = normalize(alpha*zero + beta*one)
    bell = normalize(kron(zero, zero) + kron(one, one))
    state = kron(psi, bell)
    plot_state(state, "Initial combined state |ψ> + Bell pair")
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"Initialized qubit |ψ> = {alpha}|0> + {beta}|1>\n")
    output_text.insert(tk.END, f"Normalized |ψ> = {psi.flatten()}\n")

def alice_ops():
    global state
    if state is None:
        messagebox.showinfo("Info", "Please initialize first.")
        return
    U_cnot = kron(CNOT, I)
    state = U_cnot @ state
    U_h = kron(H, I, I)
    state = U_h @ state
    plot_state(state, "After Alice's CNOT + Hadamard")
    output_text.insert(tk.END, "Alice applied CNOT and Hadamard.\n")

def measure():
    global state, alice_result
    if state is None:
        messagebox.showinfo("Info", "Initialize and run Alice's operations first.")
        return
    alice_result, state = measure_two_qubits(state)
    plot_state(state, f"After Alice measures {alice_result}")
    output_text.insert(tk.END, f"Alice measurement result: {alice_result}\n")

def bob_correction():
    global state, psi, alice_result
    if state is None or alice_result is None:
        messagebox.showinfo("Info", "Run previous steps first.")
        return
    a,b = alice_result
    if (a,b)==(0,0): correction = I
    elif (a,b)==(0,1): correction = X
    elif (a,b)==(1,0): correction = Z
    else: correction = Z @ X
    U_corr = kron(I, I, correction)
    state = U_corr @ state
    plot_state(state, "Bob's qubit after correction")
    bob_qubit = normalize(state[-2:].reshape(2,1))
    output_text.insert(tk.END, f"\nBob's final qubit:\n{bob_qubit}\n")
    output_text.insert(tk.END, f"Original |ψ>:\n{psi}\n")

# -------------------------------
# Buttons
# -------------------------------
ttk.Button(frame, text="1️⃣ Initialize", command=initialize).grid(column=0, row=1, padx=5, pady=5)
ttk.Button(frame, text="2️⃣ Alice Ops", command=alice_ops).grid(column=1, row=1, padx=5, pady=5)
ttk.Button(frame, text="3️⃣ Measure", command=measure).grid(column=2, row=1, padx=5, pady=5)
ttk.Button(frame, text="4️⃣ Bob Correction", command=bob_correction).grid(column=3, row=1, padx=5, pady=5)

root.mainloop()
