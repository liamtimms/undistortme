# undistortme runtime image
#
# Contains: undistortme (MIT), dcm2niix (BSD), slicenii (prebuilt release
# binaries), and the FSL components needed for TOPUP.
#
# NOTE ON LICENSING: FSL is installed at build time from the University of
# Oxford's own conda channel and is licensed for NON-COMMERCIAL use only
# (https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html). Commercial users must
# obtain an FSL licence from Oxford University Innovation.

FROM mambaorg/micromamba:1.5-noble

LABEL org.opencontainers.image.title="undistortme" \
      org.opencontainers.image.description="Susceptibility distortion correction for multi-echo EPI via FSL TOPUP" \
      org.opencontainers.image.source="https://github.com/liamtimms/undistortme" \
      org.opencontainers.image.licenses="MIT AND LicenseRef-FSL-NonCommercial"

# FSL pieces (topup/applytopup, fslmerge/fslmaths) from Oxford's channel,
# dcm2niix + python from conda-forge.
RUN micromamba install -y -n base \
      -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
      -c conda-forge \
      python=3.12 pip \
      fsl-topup fsl-avwutils fsl-data_standard \
      dcm2niix \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH \
    FSLDIR=/opt/conda \
    FSLOUTPUTTYPE=NIFTI

# slicenii + combinenii prebuilt release binaries (Rust, by the same author)
USER root
ADD https://github.com/liamtimms/slicenii/releases/download/0.2.0/linux-gnu.zip /tmp/slicenii.zip
RUN apt-get update && apt-get install -y --no-install-recommends unzip \
    && unzip -o /tmp/slicenii.zip -d /tmp/slicenii \
    && install -m 0755 $(find /tmp/slicenii -type f -name 'slicenii') /usr/local/bin/slicenii \
    && install -m 0755 $(find /tmp/slicenii -type f -name 'combinenii') /usr/local/bin/combinenii \
    && rm -rf /tmp/slicenii /tmp/slicenii.zip \
    && apt-get purge -y unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# The package itself
COPY --chown=mambauser:mambauser . /src/undistortme
# [test] extra included so the image can self-verify (see image.yml smoke step)
RUN pip install --no-cache-dir "/src/undistortme[test]"

USER mambauser
WORKDIR /data
ENTRYPOINT ["undistortme"]
CMD ["--help"]
