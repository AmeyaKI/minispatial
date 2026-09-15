"""Imported by ``terratorch fit --custom_modules_path train/custom_modules``.

TerraTorch imports this directory as a package before building the model, so
anything registered here is visible to the YAML. Only the UNet control lives
here (``UNetSmallFactory``); Prithvi models need nothing.
"""

from minispatial.models.unet_small import register_factory

register_factory()
