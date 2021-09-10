# osm_data_loader



## Prerequisites

* sudo pip2 install osmium
* sudo pip2 install pyshp
* sudo pip2 install utm
* sudo pip2 install mapnik
 * or follow https://github.com/mapnik/mapnik/wiki/UbuntuInstallation
   * or do
     * `git submodule update --init --recursive`
     * `cd python-mapnik`
     * `sudo apt install libpython2.7 && sudo apt install libboost-python-dev && sudo apt install clang`
     * ` ### check permission of /usr/local/lib/python3/dist-packages` 
     * `export MASON_BUILD=true && python setup.py install`

  
## Usage

* ```python2 osmFetcher/osmFetcher.py test.nikrulez pz_eifeltor.osm```


### .nikrulez 

* write feature rules to define, which object contained in the .osm map should be interpreted as which class

  * .osm differentiates between node, way and area features
  * reach feature can be defined via a python eval 
  * to access feature related keys (for example the height of a building) you can access all attributes (defined in .osm) via ```f['building:levels']```
    * in this example the feature could be expressed like this: ```building_medium: f['building'] and int(f['building:levels']) > 2```

* write out rendering rules to define, which feature should be rendered and how the rendering should be done

  * ```target``` defines the feature which should be used for rendering

  * ```define``` contains the different rendering options - these ultimately refer to [mapnik symbolizers](https://github.com/mapnik/mapnik/wiki/SymbologySupport) 

    * our example above could be rendered like this: 

      - ```target: building_medium
        target: building_medium
        define:
          fill : Color(2, 126, 255)
          fill_opacity : 0.7
          height: $(building:levels)*300.0
          draw : building 
        ```



## Todo

...

