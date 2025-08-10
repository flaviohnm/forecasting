import sys
import pkg_resources

def check_environment():
    """
    Script de diagnóstico para verificar o ambiente Python e as dependências.
    """
    print("--- INICIANDO DIAGNÓSTICO DO AMBIENTE VIRTUAL ---")
    
    # 1. Verifica a versão do Python
    print(f"\n[INFO] Versão do Python: {sys.version}")
    print(f"[INFO] Executável do Python: {sys.executable}")

    # 2. Verifica a versão da neuralforecast
    try:
        import neuralforecast
        print(f"\n[SUCESSO] Biblioteca 'neuralforecast' encontrada.")
        print(f"         Versão instalada: {neuralforecast.__version__}")
    except ImportError:
        print("\n[ERRO] Biblioteca 'neuralforecast' NÃO encontrada.")
        return

    # 3. Tenta importar NHiTS das duas formas possíveis
    print("\n--- Testando caminhos de importação para NHiTS ---")
    
    # Tentativa 1: from neuralforecast.models import NHiTS
    try:
        from neuralforecast.models import NHiTS
        print("[SUCESSO] Importação de 'from neuralforecast.models import NHiTS' FUNCIONOU.")
        print("         -> O caminho correto de importação é o genérico.")
    except ImportError as e:
        print(f"[FALHA]   Importação de 'from neuralforecast.models import NHiTS' FALHOU.")
        print(f"           Erro: {e}")

    # Tentativa 2: from neuralforecast.models.nhits import NHiTS
    try:
        from neuralforecast.models.nhits import NHiTS
        print("[SUCESSO] Importação de 'from neuralforecast.models.nhits import NHiTS' FUNCIONOU.")
        print("         -> O caminho correto de importação é o específico do submódulo.")
    except ImportError as e:
        print(f"[FALHA]   Importação de 'from neuralforecast.models.nhits import NHiTS' FALHOU.")
        print(f"           Erro: {e}")

    # 4. Lista as principais dependências instaladas
    print("\n--- Principais dependências instaladas ---")
    dependencies = ['pandas', 'numpy', 'scikit-learn', 'statsmodels', 'pmdarima', 'neuralforecast', 'pytorch-lightning']
    for lib in dependencies:
        try:
            version = pkg_resources.get_distribution(lib).version
            print(f"- {lib}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"- {lib}: NÃO INSTALADO")

    print("\n--- DIAGNÓSTICO CONCLUÍDO ---")

if __name__ == "__main__":
    check_environment()