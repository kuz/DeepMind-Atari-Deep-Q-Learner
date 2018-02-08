######################################################################
# Torch install
######################################################################

all: dependencies

TOPDIR=$(shell pwd)

# Prefix:
PREFIX=$(shell pwd)/torch

.PHONY: checkplatform all dependencies
checkplatform:
	@echo "Installing Torch into: $(PREFIX)"
ifneq ($(shell uname),Linux) 
	@echo 'Platform unsupported, only available for Linux'  && exit 1
endif
ifeq ($(strip $(shell which apt-get)),)
	@echo 'apt-get not found, platform not supported' && exit 1
endif

$(PREFIX)/.aptdependencies:
	# Install dependencies for Torch:
	sudo apt-get update
	sudo apt-get install -qqy build-essential gcc g++ cmake curl libreadline-dev git-core libjpeg-dev libpng-dev ncurses-dev imagemagick unzip libqt4-dev liblua5.1-0-dev libgd-dev 
	sudo apt-get update
	mkdir -p $(PREFIX)
	touch $@

$(PREFIX)/bin/luarocks: $(PREFIX)/.aptdependencies
	# Build and install Torch7
	rm -rf /tmp/luajit-rocks &&\
		git clone -b master https://github.com/torch/luajit-rocks.git /tmp/luajit-rocks &&\
		mkdir -p /tmp/luajit-rocks/build/ &&\
		cd /tmp/luajit-rocks/build/ &&\
		rm -f CMakeCache.txt &&\
		cmake .. -DCMAKE_INSTALL_PREFIX=$(PREFIX) -DCMAKE_BUILD_TYPE=Debug &&\
		make && make install

$(PREFIX)/lib/luarocks/rocks/%/.installed: $(PREFIX)/bin/luarocks
	$< install $*
	touch $@

LUAROCKS_TO_INSTALL:=cwrap paths torch nn luafilesystem penlight sys xlua image env qtlua qttorch nngraph mobdebug

isnvcc=$(shell which nvcc)
ifneq ($(strip $(isnvcc)),)
	LUAROCKS_TO_INSTALL:=$(LUAROCKS_TO_INSTALL) cunn cutorch
endif

LUAROCKS_TARGETS:=$(foreach rock,$(LUAROCKS_TO_INSTALL),$(PREFIX)/lib/luarocks/rocks/$(rock)/.installed)

$(PREFIX)/lib/libxitari.so: $(PREFIX)/bin/luarocks $(LUAROCKS_TARGETS)
	echo "Installing Xitari ... " &&\
	cd /tmp &&\
	rm -rf xitari &&\
	git clone https://github.com/deepmind/xitari.git &&\
	cd xitari &&\
	$(PREFIX)/bin/luarocks make
	@echo "Xitari installation completed"

$(PREFIX)/lib/lua/5.1/libalewrap.so: $(PREFIX)/bin/luarocks $(LUAROCKS_TARGETS) $(PREFIX)/lib/libxitari.so
	echo "Installing Alewrap ... " &&\
	cd /tmp &&\
	rm -rf alewrap &&\
	git clone https://github.com/deepmind/alewrap.git &&\
	cd alewrap &&\
	$(PREFIX)/bin/luarocks make
	@echo "Alewrap installation completed"

$(PREFIX)/lib/luarocks/rocks/luagd/.installed: $(PREFIX)/bin/luarocks $(LUAROCKS_TARGETS)
	echo "Installing Lua-GD ... " &&\
	mkdir -p $(PREFIX)/src &&\
	cd $(PREFIX)/src &&\
	rm -rf lua-gd &&\
	git clone https://github.com/ittner/lua-gd.git &&\
	cd lua-gd &&\
	sed -i "s/LUABIN=lua5.1/LUABIN=..\/..\/bin\/luajit/" Makefile &&\
	$(PREFIX)/bin/luarocks make &&\
	touch $@
	@echo "Lua-GD installation completed"

dependencies:  $(PREFIX)/lib/luarocks/rocks/luagd/.installed $(PREFIX)/lib/lua/5.1/libalewrap.so $(PREFIX)/lib/libxitari.so
	@echo
	@echo "You can run experiments by executing: "
	@echo 
	@echo "   ./run_cpu game_name"
	@echo
	@echo "            or   "
	@echo
	@echo "   ./run_gpu game_name"
	@echo
	@echo "For this you need to provide the rom files of the respective games (game_name.bin) in the roms/ directory"
	@echo

