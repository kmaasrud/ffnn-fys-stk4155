

# Installation

Installing Pandoc and the required extensions is easy, just type this into the command line:

**MacOS**:

    brew install pandoc pandoc-citeproc

**Ubuntu**:

    sudo apt-get pandoc pandoc-citeproc

**Arch Linux**:

    sudo pacman -S pandoc pandoc-citeproc
    
## Installing `pandoc-xnos` for cross referencing {#sec:install-xnos}
    
To get cross-referencing up and running, `pandoc-xnos` must be installed. Ensure a Python installation is present, then run:

    sudo pip install pandoc-eqnos pandoc-secnos pandoc-tablenos pandoc-fignos
    
When using cross-referencing in your document, be sure to include the option `--filter pandoc-xnos` when running Pandoc from the command line.