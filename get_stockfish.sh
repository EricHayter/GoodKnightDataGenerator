#!/bin/bash

" Downloads version 17.1 of stock fish ubuntu for x86 extracts the file and
" deletes the tarball.
wget "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-ubuntu-x86-64.tar"
tar -xf stockfish-ubuntu-x86-64.tar
rm stockfish-ubuntu-x86-64.tar
