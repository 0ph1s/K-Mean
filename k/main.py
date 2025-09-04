import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------------------
## 1. Conexão com o SQLite
db_path = "northwind.sqlite"  # ajuste se necessário
cnxn = sqlite3.connect(db_path)

#-----------------------------------------------------------------------------------------------------------------------------------
## 2. Listar tabelas
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(tables_query, cnxn)
print("Tabelas disponíveis no banco:\n")
print(tables)

# Escolha da tabela
chosen_table = input("\nDigite o nome da tabela que deseja analisar: ")

#-----------------------------------------------------------------------------------------------------------------------------------
## 3. Listar colunas da tabela escolhida
cols_query = f"PRAGMA table_info({chosen_table});"
cols = pd.read_sql(cols_query, cnxn)
print(f"\nColunas da tabela {chosen_table}:")
print(cols[['name', 'type']])

# Identificar colunas numéricas
num_cols = cols[cols['type'].str.contains("INT|REAL|NUM|DEC|DOUBLE|FLOAT", case=False)]['name'].tolist()
print(f"\nColunas numéricas detectadas: {num_cols}")

if not num_cols:
    print("⚠ Nenhuma coluna numérica encontrada nesta tabela. Encerrando.")
    exit()

# Escolha das colunas para clustering
print("\nDigite as colunas numéricas que deseja usar para clustering (separe por vírgula):")
chosen_columns = input().split(",")
chosen_columns = [c.strip() for c in chosen_columns if c.strip() in num_cols]

if not chosen_columns:
    print("⚠ Nenhuma coluna válida escolhida. Encerrando.")
    exit()

#-----------------------------------------------------------------------------------------------------------------------------------
## 4. Carregar dados da tabela
query = f"SELECT {', '.join(chosen_columns)} FROM {chosen_table}"
df = pd.read_sql(query, cnxn)
cnxn.close()

print(f"\nDados originais da tabela {chosen_table}:")
print(df.head())

#-----------------------------------------------------------------------------------------------------------------------------------
## 5. Pré-processamento
df = df.fillna(0)
data_to_cluster = df[chosen_columns]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_cluster)
scaled_df = pd.DataFrame(scaled_data, columns=chosen_columns)

print("\nDados Normalizados:")
print(scaled_df.head())

#-----------------------------------------------------------------------------------------------------------------------------------
## 6. Método do Cotovelo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title(f'Método do Cotovelo ({chosen_table})')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WCSS')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------
## 7. Aplicação do K-Means
k_ideal = int(input("\nDigite o número de clusters (k) que deseja usar: "))
kmeans = KMeans(n_clusters=k_ideal, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_data)

df['cluster'] = clusters
scaled_df['cluster'] = clusters

print("\nDataFrame com Coluna de Cluster:")
print(df.head())

#-----------------------------------------------------------------------------------------------------------------------------------
## 8. Análise dos Clusters
cluster_analysis = df.groupby('cluster').mean().reset_index()
print("\nAnálise de Cluster (Médias):")
print(cluster_analysis)

#-----------------------------------------------------------------------------------------------------------------------------------
## 9. GRÁFICOS DE DISPERSÃO MELHORADOS

# 9.1 - Gráficos de dispersão com DADOS ORIGINAIS
print("\n=== GRÁFICOS DE DISPERSÃO COM DADOS ORIGINAIS ===")

# Todas as combinações de 2 colunas
combinations_list = list(combinations(chosen_columns, 2))
n_combinations = len(combinations_list)

if n_combinations > 0:
    # Calcular layout da grade
    cols_per_row = min(3, n_combinations)  # Máximo 3 gráficos por linha
    rows = (n_combinations + cols_per_row - 1) // cols_per_row
    
    plt.figure(figsize=(6 * cols_per_row, 5 * rows))
    
    for i, (col1, col2) in enumerate(combinations_list):
        plt.subplot(rows, cols_per_row, i + 1)
        scatter = plt.scatter(df[col1], df[col2], c=df['cluster'], 
                            cmap='viridis', alpha=0.7, s=50)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'Clusters: {col1} x {col2}')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 9.2 - Gráficos de dispersão com DADOS NORMALIZADOS
print("\n=== GRÁFICOS DE DISPERSÃO COM DADOS NORMALIZADOS ===")

if n_combinations > 0:
    plt.figure(figsize=(6 * cols_per_row, 5 * rows))
    
    for i, (col1, col2) in enumerate(combinations_list):
        plt.subplot(rows, cols_per_row, i + 1)
        scatter = plt.scatter(scaled_df[col1], scaled_df[col2], c=scaled_df['cluster'], 
                            cmap='viridis', alpha=0.7, s=50)
        plt.xlabel(f'{col1} (Normalizado)')
        plt.ylabel(f'{col2} (Normalizado)')
        plt.title(f'Clusters Normalizados: {col1} x {col2}')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        # Adicionar centroids
        centroids = kmeans.cluster_centers_
        col1_idx = chosen_columns.index(col1)
        col2_idx = chosen_columns.index(col2)
        plt.scatter(centroids[:, col1_idx], centroids[:, col2_idx], 
                   c='red', marker='X', s=200, label='Centroids')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# 9.3 - Gráfico de dispersão principal com seaborn
if len(chosen_columns) >= 2:
    print("\n=== GRÁFICO DE DISPERSÃO PRINCIPAL (SEABORN) ===")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x=chosen_columns[0], y=chosen_columns[1], 
                   hue='cluster', palette='viridis', s=100, alpha=0.8)
    plt.title(f'Análise de Clusters: {chosen_columns[0]} vs {chosen_columns[1]}', 
              fontsize=16, fontweight='bold')
    plt.xlabel(chosen_columns[0], fontsize=14)
    plt.ylabel(chosen_columns[1], fontsize=14)
    plt.legend(title='Cluster', title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 9.4 - Matriz de gráficos de dispersão (Pairplot)
if len(chosen_columns) >= 2:
    print("\n=== MATRIZ DE GRÁFICOS DE DISPERSÃO (PAIRPLOT) ===")
    
    # Criar subset dos dados para pairplot
    plot_data = df[chosen_columns + ['cluster']].copy()
    
    # Pairplot com seaborn (sem criar figura manualmente)
    pair_plot = sns.pairplot(plot_data, hue='cluster', palette='viridis', 
                            plot_kws={'alpha': 0.7, 's': 50},
                            diag_kind='hist', height=3)
    pair_plot.fig.suptitle(f'Matriz de Dispersão - Tabela: {chosen_table}', 
                          fontsize=16, y=1.02)
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------
## 10. Boxplots (mantido do original)
plt.figure(figsize=(18, 6))
for i, col in enumerate(chosen_columns):
    plt.subplot(1, len(chosen_columns), i+1)
    sns.boxplot(x='cluster', y=col, data=df)
    plt.title(f'Distribuição de {col} por Cluster')

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------
## 11. Estatísticas detalhadas dos clusters
print("\n=== ESTATÍSTICAS DETALHADAS DOS CLUSTERS ===")
for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    print(f"\n--- CLUSTER {cluster_id} ---")
    print(f"Número de pontos: {len(cluster_data)}")
    print("Estatísticas:")
    print(cluster_data[chosen_columns].describe().round(2))

#-----------------------------------------------------------------------------------------------------------------------------------
## 12. Salvar CSV
df.to_csv(f'clusters_{chosen_table}.csv', index=False, encoding='utf-8')
print(f"\nDataFrame salvo como 'clusters_{chosen_table}.csv'")