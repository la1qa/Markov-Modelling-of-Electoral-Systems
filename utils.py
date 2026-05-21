import numpy as np
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

class StableMarkovChain:
    def __init__(self, df, transition_matrix=None):
        self.columns = df.columns
        self.num_columns = len(self.columns)
        self.rows = df.index
        self.num_rows = len(self.rows)

        self.df = df
        
        # Initialize as Identity matrix if none is provided
        if transition_matrix is None:
            self.transition_matrix = np.eye(self.num_columns)
        else:
            self.transition_matrix = transition_matrix

    def fit(self, decay_rate=0.1):
        """
        Fits a single, global party-to-party transition matrix across all years,
        applying exponential weights to prioritize more recent election cycles.
        """
        df = self.df
        # 1. Prepare features (X) and targets (y)
        X_all = df.iloc[:-1].values  # Vote shares at time t
        y_all = df.iloc[1:].values   # Vote shares at time t+1
        
        n_parties = self.num_columns
        num_transitions = len(X_all)
        
        # 2. Compute Exponential Time Weights
        # Create a linear sequence representing time steps [0, 1, 2, ..., total_steps]
        time_steps = np.arange(num_transitions)
        
        # Calculate raw exponential weights (the last step will equal e^0 = 1.0)
        raw_weights = np.exp(decay_rate * (time_steps - time_steps[-1]))
        
        # Normalize weights so they sum up to the total number of transitions
        # This keeps the scale of our loss function consistent
        time_weights = (raw_weights / np.sum(raw_weights)) * num_transitions

        # --- Objective Function (Weighted MSE Loss) ---
        def objective(P_flat):
            P = P_flat.reshape((n_parties, n_parties))
            y_pred = X_all @ P
            
            # Calculate squared errors for each transition step
            squared_errors = np.mean((y_all - y_pred) ** 2, axis=1)
            
            # Apply our time weights row-by-row (element-wise multiplication)
            weighted_errors = squared_errors * time_weights
            
            return np.mean(weighted_errors)
        
        # --- Constraints & Bounds (Markov Probability Rules) ---
        constraints = []
        for r in range(n_parties):
            constraints.append({
                'type': 'eq',
                'fun': lambda P_flat, r=r: np.sum(P_flat.reshape((n_parties, n_parties))[r, :]) - 1.0
            })
        
        bounds = [(0, 1) for _ in range(n_parties * n_parties)]
        
        # --- Initial Condition: Identity Matrix (I) ---
        init_P = np.eye(n_parties)
        
        # Run the optimization solver using SLSQP
        res = minimize(objective, init_P.flatten(), method='SLSQP', bounds=bounds, constraints=constraints)
        
        if res.success:
            optimized_matrix = res.x.reshape((n_parties, n_parties))
            self.transition_matrix = pd.DataFrame(optimized_matrix, index=self.columns, columns=self.columns)
            print(f"Model successfully fitted using time weights (Decay Rate: {decay_rate})")
            print(f"Applied weights per step (oldest to newest): {np.round(time_weights, 2)}")
        else:
            print("Warning: Optimization failed to converge.")

    def get_steady_state(self):
        if isinstance(self.transition_matrix, pd.DataFrame):
            P = self.transition_matrix.values
        else:
            P = self.transition_matrix

        # FIX: Transpose P to look for LEFT eigenvectors (pi @ P = pi)
        eigenvalues, eigenvectors = np.linalg.eig(P.T)

        # Find the index of the eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1))

        # Extract the vector and normalize (sum = 1)
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()

        steady_state = pd.Series(pi, index=self.columns)
        self.steady_state = steady_state
        return steady_state

    def __str__(self):
        output = f"Stable Markov Chain Model ({self.num_rows} Years, {self.num_columns} Parties)\n"
        output += "=======================================================\n"
        output += "Global Transition Matrix (Rows = From Party, Columns = To Party):\n"
        if isinstance(self.transition_matrix, pd.DataFrame):
            output += self.transition_matrix.round(4).to_string()
        else:
            output += str(np.round(self.transition_matrix, 4))
        return output
    
    def plot_transition_matrix(self):
        plt.figure(figsize=(6, 4))
        sns.heatmap(self.transition_matrix, annot=True, cmap="Blues", cbar=False, linewidths=0.5)
        plt.title("Matriu de Transició")
        plt.show()

    def plot_steady_state(self):
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Ensure steady state is calculated and stored
        if not hasattr(self, 'steady_state') or self.steady_state is None:
            self.steady_state = self.get_steady_state()
            
        total_distribution = self.steady_state
        
        # 2. Extract Abstenció details before dropping it
        if "Abstenció" in total_distribution.index:
            abstencio_share = total_distribution["Abstenció"]
            active_parties = total_distribution.drop("Abstenció")
        else:
            abstencio_share = 0.0
            active_parties = total_distribution

        # 3. SORT descending to order largest to smallest from left to right
        active_parties = active_parties.sort_values(ascending=False)

        # 4. Renormalize the active parties so they sum to 1.0
        active_sum = active_parties.sum()
        if active_sum == 0:
            print("Error: No active party votes to plot.")
            return
            
        normalized_shares = (active_parties / active_sum).values.tolist()
        active_labels = active_parties.index.tolist()
        original_shares = active_parties.values.tolist()
        
        # 5. Construct the data for the half-pie chart
        shares_with_dummy = normalized_shares + [1.0]
        
        # 6. --- DYNAMIC COLORSET GENERATION ---
        num_active = len(normalized_shares)
        
        # Using 'viridis' as a base, but you can swap to 'mako', 'plasma', 'crest', or 'GnBu'
        # We sample the colormap from 0.2 to 0.85 so the colors stay vibrant (avoiding pure white or pure black)
        cmap = plt.cm.get_cmap('viridis')
        active_colors = cmap(np.linspace(0.2, 0.85, num_active))
        
        # Append a completely transparent RGBA color for the hidden bottom-half slice
        colors_with_dummy = list(active_colors) + [(0, 0, 0, 0)]
        
        # 7. Build dual-percentage custom labels
        custom_labels = []
        for label, norm_share, orig_share in zip(active_labels, normalized_shares, original_shares):
            custom_labels.append(
                f"{label}\n" 
                # f"{norm_share * 100:.1f}% de vots vàlids\n"
                f"({orig_share * 100:.1f}% del cens)"
                if norm_share > 0.1 else 
                f"{label} "
                # f"{norm_share * 100:.1f}% de vots vàlids\n"
                f"({orig_share * 100:.1f}% del cens)"
            )
            
        custom_labels += [""]

        # 8. Plotting with enhanced styling
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['text.color'] = '#333333'
        
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        
        # Clean white borders between the color-mapped slices
        ax.pie(
            shares_with_dummy, 
            labels=custom_labels, 
            autopct='', 
            startangle=180, 
            counterclock=False,
            colors=colors_with_dummy,
            labeldistance=1.3,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
            textprops={'fontsize': 9.0, 'color': '#333333'}
        )
        
        # Crop to show only the top semicircle
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(0, 1.35)
        
        # Footnote displaying the extracted abstention data in a muted gray
        if abstencio_share > 0:
            plt.text(
                0, -0.15, 
                f"*'Abstenció' ({abstencio_share*100:.1f}% del cens total)", 
                ha='center', va='center', fontsize=9.5, style='italic', color='#666666'
            )
        
        plt.title("Solució Estable", pad=25, fontsize=13, weight='bold', color='#1A1A1A')
        plt.tight_layout()
        plt.show()
        
    def verify_steady_state(self):
        """
        Verifica les hipòtesis d'existència d'una única distribució estacionària.
        """
        if isinstance(self.transition_matrix, pd.DataFrame):
            P = self.transition_matrix.values
        else:
            P = self.transition_matrix

        n = P.shape[0]

        print("VERIFICACIÓ DE L'EXISTÈNCIA DE LA DISTR. ESTACIONÀRIA")

        # ── 1. Matriu estocàstica ──
        sumes_files = P.sum(axis=1)
        files_sumen_1 = np.allclose(sumes_files, 1)
        valors_no_negatius = np.all(P >= -1e-10)
        print(f"\n1. Matriu estocàstica:")
        print(f"   · Sumes de cada fila: {np.round(sumes_files, 6)}")
        print(f"   · Files sumen 1:      {files_sumen_1}")
        print(f"   · Valors no negatius: {valors_no_negatius}")
        es_estocàstica = files_sumen_1 and valors_no_negatius

        if not es_estocàstica:
            print("\nNo és matriu estocàstica. Para.")
            return False

        # ── 2. Irreductibilitat ──
        reach = np.eye(n) + P
        reach_pow = np.linalg.matrix_power(reach, n - 1)
        irreductible = np.all(reach_pow > 1e-10)
        print(f"\n2. Irreductibilitat (tots els estats es comuniquen): {irreductible}")

        if not irreductible:
            print("\nNo és irreductible. Para.")
            return False

        # ── 3. Aperiodicitat ──
        te_self_loops = np.any(np.diag(P) > 0)
        print(f"\n3. Aperiodicitat (self-loops a la diagonal): {te_self_loops}")
        if te_self_loops:
            print(f"   → La cadena és APERIÒDICA")
        else:
            print(f"   → Cal anàlisi addicional del GCD dels cicles")

        if not te_self_loops:
            print("\nNo es pot garantir aperiodicitat.")
            return False

        # ── Veredicte ──
        print("\nEXISTEIX distribució estacionària ÚNICA")
        return True


if __name__ == "__main__":
    # Stable election data (No NaNs, same parties)
    data = {
        'Party_A': [0.40, 0.38, 0.35, 0.36],
        'Party_B': [0.35, 0.36, 0.38, 0.37],
        'Party_C': [0.25, 0.26, 0.27, 0.27]
    }
    df = pd.DataFrame(data, index=[2012, 2016, 2020, 2024])
    
    print("Stable Election Data:")
    print(df, "\n")
    
    mc = StableMarkovChain(df)
    mc.fit()
    print(mc)

    print("\nSteady State Distribution:")
    print(mc.get_steady_state())
    mc.plot_transition_matrix()
    mc.plot_steady_state()