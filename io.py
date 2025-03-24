import xarray as xr


# Load nc files
def load_nc(path):
    try:
        return xr.open_mfdataset(
            path,
            concat_dim="tomo_zdim",
            data_vars= "minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )
    except:
        return xr.open_mfdataset(
            path,
            concat_dim="labels_zdim",
            data_vars= "minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )

def load_nc_arr(path):
    try:
        return xr.open_mfdataset(
            path,
            concat_dim="tomo_zdim",
            data_vars= "minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )['tomo'].data
    except:
        return xr.open_mfdataset(
            path,
            concat_dim="labels_zdim",
            data_vars= "minimal",
            combine="nested",
            combine_attrs="drop_conflicts",
            coords="minimal",
            compat="override",
        )['labels'].data