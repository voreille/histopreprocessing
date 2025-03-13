def wsi_id_mapping_tcga(wsi_filename):
    return wsi_filename.split(".")[0]


def wsi_id_mapping_cptac(wsi_filename):
    return wsi_filename.split(".")[0]


WSI_ID_MAPPING_DICT = {
    "TCGA": wsi_id_mapping_tcga,
    "CPTAC": wsi_id_mapping_cptac,
}
