import numpy as np
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

class StableMarkovChain:
    def __init__(self, df, transition_matrix=None):
        self.columns = df.columns
        self.num_columns = len(self.columns)
        self.rows = df.index
        self.num_rows = len(self.rows)

        self.df = df
        
        # Inicialitzem com a matriu identitat si no es proporciona cap
        if transition_matrix is None:
            self.transition_matrix = np.eye(self.num_columns)
        else:
            self.transition_matrix = transition_matrix

    def fit(self, decay_rate=0.2, sparse_threshold=1e-9, fidelitat_threshold=0.001):
        """
        Ajusta una matriu de transició global partit-a-partit per tots els anys,
        aplicant pesos exponencials per prioritzar els cicles electorals més recents.
        Convenció: columnes sumen 1 (x_{t+1} = P @ x_t)
        """
        df = self.df

        # 1. Preparem les dades (X = temps t, y = temps t+1)
        X_all = df.iloc[:-1].values  # Distribució de vots en el temps t
        y_all = df.iloc[1:].values   # Distribució de vots en el temps t+1
        
        n_parties = self.num_columns
        num_transitions = len(X_all)
        
        # 2. Calculem els pesos temporals exponencials
        # Seqüència lineal de passos temporals [0, 1, 2, ..., total_passos]
        time_steps = np.arange(num_transitions)
        
        # Pesos exponencials bruts (l'últim pas valdrà e^0 = 1.0)
        raw_weights = np.exp(decay_rate * (time_steps - time_steps[-1]))
        
        # Normalitzem els pesos perquè sumin el nombre total de transicions
        # Això manté l'escala de la funció de pèrdua consistent
        time_weights = (raw_weights / np.sum(raw_weights)) * num_transitions

        # --- Funció objectiu (MSE ponderada) ---
        def objective(P_flat):
            P = P_flat.reshape((n_parties, n_parties))
            
            # Convenció columna suma 1: y = (P @ x.T).T
            y_pred = (P @ X_all.T).T
            
            # Errors quadràtics per cada pas temporal
            squared_errors = np.mean((y_all - y_pred) ** 2, axis=1)
            
            # Apliquem els pesos temporals fila a fila
            weighted_errors = squared_errors * time_weights

            # Terme de fidelitat: penalitza allunyar-se de la identitat
            # Valor baix (0.01) perquè tenim moltes dades i no cal molta regularització
            fidelitat = fidelitat_threshold * np.mean((P - np.eye(n_parties)) ** 2)
            return np.mean(weighted_errors) + fidelitat
            
        # --- Restriccions i límits (regles de probabilitat de Markov) ---
        constraints = []
        for r in range(n_parties):
            # Convenció columna suma 1: les columnes han de sumar 1
            constraints.append({
                'type': 'eq',
                'fun': lambda P_flat, r=r: np.sum(P_flat.reshape((n_parties, n_parties))[:, r]) - 1.0
            })
        
        # Tots els valors de P entre 0 i 1
        bounds = [(0, 1) for _ in range(n_parties * n_parties)]
        
        # --- Condició inicial: Matriu identitat (I) ---
        init_P = np.eye(n_parties)
        
        # Executem l'optimitzador SLSQP
        res = minimize(objective, init_P.flatten(), method='SLSQP', bounds=bounds, constraints=constraints)
     
        if res.success:
            optimized_matrix = res.x.reshape((n_parties, n_parties))
            
            # Apliquem threshold: valors menors que el llindar es posen a 0
            optimized_matrix[optimized_matrix < sparse_threshold] = 0
            
            # Renormalitzem columnes perquè segueixin sumant 1
            col_sums = optimized_matrix.sum(axis=0, keepdims=True)
            optimized_matrix = optimized_matrix / col_sums
            
            self.transition_matrix = pd.DataFrame(optimized_matrix, index=self.columns, columns=self.columns)
            print(f"Model ajustat correctament amb pesos temporals (Decay Rate: {decay_rate})")
            print(f"Pesos aplicats per pas (de més antic a més recent): {np.round(time_weights, 2)}")
        else:
            print("Avís: L'optimització no ha convergit.")
            
    def get_steady_state(self):
        """
        Calcula la distribució estacionària π de forma robusta.
        """
        P = self.transition_matrix.values if isinstance(self.transition_matrix, pd.DataFrame) else self.transition_matrix
        n = P.shape[0]

        # 1. Crear el graf explícitament com a dirigit
        G = nx.DiGraph()
        for i in range(n):
            for j in range(n):
                if P[i, j] > 0:
                    G.add_edge(i, j) # Afegim les connexions (i -> j)

        # 2. Comprovar la connectivitat forta en un graf dirigit
        is_irreducible = nx.is_strongly_connected(G)
        
        # 3. Mapeig d'etiquetes segur
        labels = self.columns if (isinstance(self.transition_matrix, pd.DataFrame) and len(self.columns) == n) else [f"Estat_{i}" for i in range(n)]

        if is_irreducible:
            # CAS IRREDUCTIBLE
            eigenvalues, eigenvectors = np.linalg.eig(P)
            idx = np.argmin(np.abs(eigenvalues - 1))
            pi = np.real(eigenvectors[:, idx])
            pi = pi / pi.sum()
            steady_state = pd.Series(pi, index=labels)
        else:
            print("La matriu és reductible. Calculant comportament a llarg termini...")
            P_n = np.linalg.matrix_power(P, 1000)
            steady_state_avg = P_n.mean(axis=1) 
            steady_state = pd.Series(steady_state_avg, index=labels)
            
        self.steady_state = steady_state
        return steady_state
    """def get_steady_state(self):
        
        Calcula la distribució estacionària π tal que P @ π = π.
        Convenció columna suma 1: busquem el vector propi DRET de P associat a λ=1.
    
        if isinstance(self.transition_matrix, pd.DataFrame):
            P = self.transition_matrix.values
        else:
            P = self.transition_matrix
        
        # 2. Comprovem irreductibilitat
        G = nx.DiGraph()
        n = P.shape[0]
        for i in range(n):
            for j in range(n):
                if P[i, j] > 0:
                    G.add_edge(j, i)
        is_irreducible = nx.is_strongly_connected(G)
        
        if is_irreducible:

            # Convenció columna: busquem vectors propis DRETS (P @ v = v)
            eigenvalues, eigenvectors = np.linalg.eig(P)

            # Trobem l'índex del valor propi més proper a 1
            idx = np.argmin(np.abs(eigenvalues - 1))

            # Extraiem el vector i normalitzem (suma = 1)
            pi = np.real(eigenvectors[:, idx])
            pi = pi / pi.sum()

            steady_state = pd.Series(pi, index=self.columns)
            self.steady_state = steady_state
            return steady_state
        else:
            print("La matriu no és irreductible. No es pot garantir una distribució estacionària única.")
            print("Calculant el comportament a llarg termini buscant classes absorbents...")
            P_n = np.linalg.matrix_power(P, 1000)
            steady_state_avg = P_n.mean(axis=1)  
            steady_state = pd.Series(steady_state_avg, index=self.columns)           
            self.steady_state = steady_state
            return steady_state"""

    def __str__(self):
        output = f"Model de Cadena de Markov Estable ({self.num_rows} Anys, {self.num_columns} Partits)\n"
        output += "=======================================================\n"
        output += "Matriu de Transició Global (Files = Destí, Columnes = Origen):\n"
        if isinstance(self.transition_matrix, pd.DataFrame):
            output += self.transition_matrix.round(4).to_string()
        else:
            output += str(np.round(self.transition_matrix, 4))
        return output
    
    def plot_transition_matrix(self):
        """Visualitza la matriu de transició com a heatmap."""
        plt.figure(figsize=(6, 4))
        sns.heatmap(self.transition_matrix, annot=True, cmap="Blues", cbar=False, linewidths=0.5)
        plt.title("Matriu de Transició")
        plt.show()

    def plot_steady_state(self):
        """Visualitza la distribució estacionària com a semicercle."""

        # 1. Calculem la distribució estacionària si no existeix
        if not hasattr(self, 'steady_state') or self.steady_state is None:
            self.steady_state = self.get_steady_state()
            
        total_distribution = self.steady_state
        
        # 2. Extraiem Abstenció abans d'eliminar-la del gràfic principal
        if "Abstenció" in total_distribution.index:
            abstencio_share = total_distribution["Abstenció"]
            active_parties = total_distribution.drop("Abstenció")
        else:
            abstencio_share = 0.0
            active_parties = total_distribution

        # 3. Ordenem de major a menor (d'esquerra a dreta)
        active_parties = active_parties.sort_values(ascending=False)

        # 4. Renormalitzem els partits actius perquè sumin 1.0
        active_sum = active_parties.sum()
        if active_sum == 0:
            print("Error: No hi ha vots de partits actius per mostrar.")
            return
            
        normalized_shares = (active_parties / active_sum).values.tolist()
        active_labels = active_parties.index.tolist()
        original_shares = active_parties.values.tolist()
        
        # 5. Afegim la porció dummy per tancar el semicercle
        shares_with_dummy = normalized_shares + [1.0]
        
        # 6. Generem colors dinàmicament amb el mapa 'viridis'
        # Mostregem entre 0.2 i 0.85 per evitar colors massa clars o foscos
        num_active = len(normalized_shares)
        cmap = plt.cm.get_cmap('viridis')
        active_colors = cmap(np.linspace(0.2, 0.85, num_active))
        
        # Color completament transparent per la meitat inferior oculta
        colors_with_dummy = list(active_colors) + [(0, 0, 0, 0)]
        
        # 7. Construïm les etiquetes amb doble percentatge
        custom_labels = []
        for label, norm_share, orig_share in zip(active_labels, normalized_shares, original_shares):
            custom_labels.append(
                f"{label}\n" 
                f"({orig_share * 100:.1f}% del cens)"
                if norm_share > 0.1 else 
                f"{label} "
                f"({orig_share * 100:.1f}% del cens)"
            )
            
        custom_labels += [""]

        # 8. Dibuixem el gràfic amb estil refinat
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['text.color'] = '#333333'
        
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        
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
        
        # Retallem per mostrar només el semicercle superior
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(0, 1.35)
        
        # Nota a peu de pàgina amb les dades d'abstenció
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
        Verifica les hipòtesis d'existència d'una única distribució estacionària
        mitjançant el Teorema de Perron-Frobenius.
        """
        if isinstance(self.transition_matrix, pd.DataFrame):
            P = self.transition_matrix.values
        else:
            P = self.transition_matrix

        n = P.shape[0]

        print("VERIFICACIÓ DE L'EXISTÈNCIA DE LA DISTR. ESTACIONÀRIA")

        # ── 1. Matriu estocàstica per columnes ──
        sumes_columnes = P.sum(axis=0)
        columnes_sumen_1 = np.allclose(sumes_columnes, 1)
        valors_no_negatius = np.all(P >= -1e-10)
        print(f"\n1. Matriu estocàstica (columnes sumen 1):")
        print(f"   · Sumes de cada columna: {np.round(sumes_columnes, 6)}")
        print(f"   · Columnes sumen 1:      {columnes_sumen_1}")
        print(f"   · Valors no negatius:    {valors_no_negatius}")
        es_estocàstica = columnes_sumen_1 and valors_no_negatius

        if not es_estocàstica:
            print("\nNo és matriu estocàstica.")
            return False

        # ── 2. Irreductibilitat mitjançant graf dirigit (networkx) ──
        # Usem networkx per evitar problemes numèrics del mètode matricial
        G = nx.DiGraph()
        for i in range(n):
            for j in range(n):
                if P[i, j] > 0:
                    G.add_edge(i, j)

        irreductible = nx.is_strongly_connected(G)
        print(f"\n2. Irreductibilitat (tots els estats es comuniquen): {irreductible}")

        if not irreductible:
            print("\nNo és irreductible. ÉS REDUCTIBLE → No es pot garantir una distribució estacionària única.")
            return False

        # ── 3. Aperiodicitat: self-loops a la diagonal ──
        te_self_loops = np.any(np.diag(P) > 0)
        print(f"\n3. Aperiodicitat (self-loops a la diagonal): {te_self_loops}")
        if te_self_loops:
            print(f"   → La cadena és APERIÒDICA")
        else:
            print(f"   → Cal anàlisi addicional del MCD dels cicles")

        if not te_self_loops:
            print("\nNo es pot garantir aperiodicitat.")
            return False

        # ── Veredicte final ──
        print("\nEXISTEIX distribució estacionària ÚNICA")
        return True



    def plot_graph(self):
        """
        Visualitza el graf amb self-loops clars, fletxes visibles i estètica neta.
        """

        if isinstance(self.transition_matrix, pd.DataFrame):
            P = self.transition_matrix.values
            labels = list(self.transition_matrix.columns)
        else:
            P = self.transition_matrix
            labels = [f"P{i}" for i in range(len(P))]

        G = nx.DiGraph()
        for i, row in enumerate(labels):
            for j, col in enumerate(labels):
                if P[i, j] > 1e-9:
                    G.add_edge(row, col, weight=P[i, j])

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        pos = nx.spring_layout(G, k=1.2, seed=42)

        # Paleta de colors — un per node/origen
        palette = [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B7A57',
            '#7B2D8B', '#44BBA4', '#E94F37', '#F5A623', '#1B4F72', '#117A65'
        ]
        node_colors = {node: palette[i % len(palette)] for i, node in enumerate(G.nodes())}

        node_radius = 0.07

        # ── Nodes ──────────────────────────────────────────────────────────────
        for node, (x, y) in pos.items():
            circle = plt.Circle((x, y), node_radius,
                                color=node_colors[node],
                                zorder=5, linewidth=2,
                                edgecolor='white')
            ax.add_patch(circle)
            ax.text(x, y, str(node),
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white', zorder=6,
                    fontfamily='monospace')

        # ── Edges ──────────────────────────────────────────────────────────────
        edge_label_positions = {}

        for (u, v, data) in G.edges(data=True):
            w = data['weight']
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            color = node_colors[u]  # color del node origen

            # ── Self-loop ──────────────────────────────────────────────────────
            if u == v:
                loop_radius = 0.12
                theta = np.pi / 2
                neighbors = list(G.predecessors(u)) + list(G.neighbors(u))
                angles = []
                for nb in neighbors:
                    if nb != u:
                        nx_, ny_ = pos[nb]
                        angles.append(np.arctan2(ny_ - y1, nx_ - x1))
                if angles:
                    angles_r = sorted(set([round(a, 1) for a in angles]))
                    candidates = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                    best = max(candidates,
                            key=lambda c: min(abs(c - a) for a in angles_r))
                    theta = best

                cx = x1 + loop_radius * np.cos(theta)
                cy = y1 + loop_radius * np.sin(theta)
                loop = plt.Circle((cx, cy), loop_radius * 0.85,
                                fill=False, color=color,
                                linewidth=1.8, zorder=3)
                ax.add_patch(loop)

                angle_arrow = theta + np.pi + 0.3
                ax_tip = cx + loop_radius * 0.85 * np.cos(angle_arrow)
                ay_tip = cy + loop_radius * 0.85 * np.sin(angle_arrow)
                ax.annotate('', xy=(ax_tip, ay_tip),
                            xytext=(ax_tip - 0.001 * np.cos(angle_arrow + np.pi / 2),
                                    ay_tip - 0.001 * np.sin(angle_arrow + np.pi / 2)),
                            arrowprops=dict(arrowstyle='-|>', color=color,
                                            lw=1.8, mutation_scale=18),
                            zorder=4)

                lx = cx + loop_radius * 1.1 * np.cos(theta)
                ly = cy + loop_radius * 1.1 * np.sin(theta)
                edge_label_positions[(u, v)] = (lx, ly)

            # ── Edge normal ────────────────────────────────────────────────────
            else:
                rad = 0.25 if G.has_edge(v, u) else 0.0
                angle = np.arctan2(y2 - y1, x2 - x1)
                sx = x1 + node_radius * np.cos(angle)
                sy = y1 + node_radius * np.sin(angle)
                ex = x2 - node_radius * np.cos(angle)
                ey = y2 - node_radius * np.sin(angle)

                style = f"arc3,rad={rad}"
                ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle='-|>',
                                            color=color,
                                            lw=1.4,
                                            connectionstyle=style,
                                            mutation_scale=20),
                            zorder=3)

                if rad != 0.0:
                    mx = (sx + ex) / 2 + rad * 0.5 * np.sin(angle) * (-1 if rad > 0 else 1)
                    my = (sy + ey) / 2 - rad * 0.5 * np.cos(angle) * (-1 if rad > 0 else 1)
                else:
                    mx = (sx + ex) / 2
                    my = (sy + ey) / 2
                edge_label_positions[(u, v)] = (mx, my)

        # ── Labels dels pesos ──────────────────────────────────────────────────
        for (u, v), (lx, ly) in edge_label_positions.items():
            w = G[u][v]['weight']
            ax.text(lx, ly, f"{w:.3g}",
                    fontsize=7, ha='center', va='center',
                    fontfamily='monospace', color='#222222',
                    bbox=dict(facecolor='white', edgecolor='#cccccc',
                            boxstyle='round,pad=0.25', alpha=0.95, linewidth=0.8),
                    zorder=7)

        # ── Títol i acabats ────────────────────────────────────────────────────
        ax.set_title("Graf de Transicions · Estructura de l'Oligopoli",
                    fontsize=14, fontweight='bold', color='#1a1a1a',
                    pad=16, fontfamily='monospace')
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    # Dades electorals d'exemple (sense NaNs, mateixos partits)
    data = {
        'Party_A': [0.40, 0.38, 0.35, 0.36],
        'Party_B': [0.35, 0.36, 0.38, 0.37],
        'Party_C': [0.25, 0.26, 0.27, 0.27]
    }
    df = pd.DataFrame(data, index=[2012, 2016, 2020, 2024])
    
    print("Dades electorals:")
    print(df, "\n")
    
    mc = StableMarkovChain(df)
    mc.fit()
    print(mc)

    print("\nDistribució Estacionària:")
    print(mc.get_steady_state())
    mc.plot_transition_matrix()
    mc.plot_steady_state()
    mc.plot_graph()
    
    