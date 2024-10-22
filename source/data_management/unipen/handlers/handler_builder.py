import os

from source.data_management.unipen.handlers.base_handler import UnipenHandler
from source.data_management.unipen.handlers.multi_doc_handler import MultiDocFileHandler
from source.data_management.unipen.handlers.single_doc_handler import SingleDocFileHandler
from source.data_management.unipen.handlers.no_doc_handler import NoDocFileHandler

class UnipenHandlerBuilder():
    """This class holds the knowledge of the UNIPEN data providers and their different formats.
    It can be used to build the different online signals through builders adapted to providers."""
    unipen_root: str

    provider_type_mapping = {
        SingleDocFileHandler: [
                                "abm", "anj", "apa", "apb", "apc", "apd", "ape", "app", "att", 
                               "bba", "bbb", "bbc", "bbd", 
                               "cea", "ceb", "cec", "ced", "cee",
                               "dar", "gmd", "imt", "int", 
                               "lex", "par", "pcl", "pri",
                               "rim", "scr", "uqb"
                            ],
        MultiDocFileHandler: [
                                "hpb", "hpp", "huj", "tos",
                                "kai", "kar", "lav", "lou", 
                                "mot", "pap", "phi", "sta", 
                                "syn", "val", "ugi"
                            ],
                            
        NoDocFileHandler: [
                                "art", "aga", "atu", "nic", 
                                "sie"
                            ],
        # Problems
        #   Empty: ata, cef, ibm, imp
        #   No temporal info: not

        #Par: .inc file changed to .doc and COORD added to it, as every stroke file has the same COORD system. Avoids creating a single handler for this one.
        #PCL: Rename internal_pad.doc to .doc.other as it has the same information as the pcl.doc file.
        #phi: Renaned *file* to *file*.doc
        #HPP: rename hpp doc files form hpb* into hpp*
        #HUJ: Put manudo datafile into huj8/ folder in order to have corresponding doc file
        #STA: rename hpb{0,1}.doc to sta{0,1}.doc
        #Val: Separate two writers into val01, val02 folders to correspond to expected structure
    }

    def __init__(self, unipen_root: str) -> None:
        self.unipen_root = unipen_root

    def build_handlers(self) -> list[UnipenHandler]:
        """Build all the handlers declared by this provider maping"""
        provider_return: list[UnipenHandler] = []
        
        for handler_class, providers in self.provider_type_mapping.items():
            total_providers = len(providers)
            print(f"Building handlers of type {handler_class.__name__} ({total_providers})")
            ind = 1
            for provider in providers:
                try:
                    print(f"\rBuilding provider {provider} [{ind}/{total_providers}]", end="")
                    ind += 1
                    provider_path = os.path.join(self.unipen_root, provider)
                    provider_handler = handler_class(provider_path)
                    provider_return.append(provider_handler)
                except Exception as e:
                    print(f"Impossible to build provider {provider} due to error: {e}")
                    raise e
            print()

        return provider_return