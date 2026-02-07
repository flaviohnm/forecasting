import yaml
import os

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_and_organize_strategies(yaml_path):
    config = load_config(yaml_path)
    
    if 'strategies' not in config or 'full_comparison' not in config['strategies']:
        raise KeyError("O arquivo model_params.yaml deve conter 'strategies' -> 'full_comparison'")

    strategies = config['strategies']['full_comparison']
    
    base_models = []
    hybrid_models = []

    for model in strategies:
        # --- CORREÇÃO DEFINITIVA ---
        # Não adicionamos mais prefixos ou sufixos automáticos.
        # O nome do arquivo será exatamente o 'model_name' do YAML.
        model['unique_exec_name'] = model['model_name']
        
        if 'depends_on' in model:
            hybrid_models.append(model)
        else:
            base_models.append(model)
            
    return base_models, hybrid_models