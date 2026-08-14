import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(csv_path, save_path, model_name="Modelo"):
    """
    Lee un archivo CSV con el historial de entrenamiento y genera un gráfico resumen.
    Es agnóstico y busca columnas comunes de loss y métricas para ser reutilizado
    tanto en YOLO como en modelos PyTorch personalizados.
    """
    if not os.path.exists(csv_path):
        return False
        
    try:
        # Leer el CSV y limpiar espacios en las columnas
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Buscar columna de epoch (eje X)
        if 'epoch' in df.columns:
            epochs = df['epoch']
        else:
            epochs = range(1, len(df) + 1)
            
        # Graficar Loss
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        
        # Búsqueda dinámica de columnas de Loss (YOLO usa train/box_loss, PyTorch puede usar train_loss)
        train_loss_col = next((col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower()), None)
        val_loss_col = next((col for col in df.columns if 'val' in col.lower() and 'loss' in col.lower()), None)
        
        if train_loss_col:
            ax1.plot(epochs, df[train_loss_col], color='tab:red', label=f'Train ({train_loss_col})', linestyle='-')
        if val_loss_col:
            ax1.plot(epochs, df[val_loss_col], color='darkred', label=f'Val ({val_loss_col})', linestyle='--')
            
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # Graficar Métricas (mAP) en otro eje Y
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Métrica Principal', color='tab:blue')  
        
        # Búsqueda dinámica de la métrica principal (priorizar mAP50 o similar)
        metric_col = next((col for col in df.columns if 'map50' in col.lower()), None)
        if not metric_col:
            metric_col = next((col for col in df.columns if 'map' in col.lower() or 'acc' in col.lower()), None)
            
        if metric_col:
            ax2.plot(epochs, df[metric_col], color='tab:blue', label=f'Metric ({metric_col})', linewidth=2)
            
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        fig.tight_layout()
        plt.title(f"Resumen de Entrenamiento - {model_name}")
        
        # Combinar leyendas en la parte inferior
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        if lines_1 or lines_2:
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        return True
        
    except Exception as e:
        print(f"Error al generar gráfico: {e}")
        return False
