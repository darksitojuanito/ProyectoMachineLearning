from src.utils import setup_logging, get_logger
from src.dblp_extractor import DBLPExtractor
from src.ieee_extractor import IEEEExtractor
from src.dataset_builder import DatasetBuilder
from src.validator import Validator

def main():
    setup_logging()
    logger = get_logger("Main")
    
    logger.info("=== INICIANDO EXTRACCIÓN DATASET ICMLA 2020 ===")
    
    # 1. Extraer de DBLP
    dblp = DBLPExtractor()
    dblp.extract()
    
    # 2. Extraer de IEEE Xplore
    ieee = IEEEExtractor()
    ieee.extract_all()
    
    # 3. Construir Datasets
    builder = DatasetBuilder()
    builder.build()
    
    # 4. Validar y Reportar
    validator = Validator()
    validator.validate()
    
    logger.info("=== PROCESO COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()
