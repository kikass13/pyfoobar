import overpy
import fiona 
from fiona.crs import from_epsg
import mapnik


api = overpy.Overpass()
# fetch all ways and nodes

south_bound = 50.8828300
west_bound = 6.9198700 
north_bound = 50.8866800
east_bound = 6.9244800

result = api.query("""
way(50.8828300,6.9198700 ,50.8866800,6.9244800) ["building"];
(._;>;);
out body;
""")
#minlat="50.8828300" minlon="6.9198700" maxlat="50.8866800" maxlon="6.9244800"

schema = {'geometry': 'LineString', 'properties': {'Name':'str:80'}}
shapeout = "test.shp"

with fiona.open(shapeout, 'w',crs=from_epsg(3857),driver='ESRI Shapefile', schema=schema) as output:
    for i, way in enumerate(result.ways):                
        #if (node.lat<north_bound and node.lat>south_bound and node.lon<west_bound and node.lon>east_bound ):
        # the shapefile geometry use (lon,lat) 
        line = {'type': 'LineString', 'coordinates':[(node.lon, node.lat) for node in way.nodes if (node.lat<north_bound and node.lat>south_bound and node.lon>west_bound and node.lon<east_bound )]}
        prop = {'Name': way.tags.get("name", "n/a")}
        output.write({'geometry': line, 'properties':prop})



m = mapnik.Map(1097,907) # create a map with a given width and height in pixels
# note: m.srs will default to '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
# the 'map.srs' is the target projection of the map and can be whatever you wish 

# Set up projections
# spherical mercator (most common target map projection of osm data imported with osm2pgsql)
merc = mapnik.Projection('+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +no_defs +over')

# long/lat in degrees, aka ESPG:4326 and "WGS 84" 
longlat = mapnik.Projection('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

m.srs = merc.params()

m.background = mapnik.Color('steelblue') # set background colour to 'steelblue'. 

s = mapnik.Style() # style object to hold rules
r = mapnik.Rule() # rule object to hold symbolizers
# to fill a polygon we create a PolygonSymbolizer
polygon_symbolizer = mapnik.PolygonSymbolizer()
polygon_symbolizer.fill = mapnik.Color('#f2eff9')  
r.symbols.append(polygon_symbolizer) # add the symbolizer to the rule object

# to add outlines to a polygon we create a LineSymbolizer
line_symbolizer = mapnik.LineSymbolizer()
line_symbolizer.stroke = mapnik.Color('green')
line_symbolizer.stroke_width = 0.5
line_symbolizer.fill = mapnik.Color('green') 
r.symbols.append(line_symbolizer) # add the symbolizer to the rule object
s.rules.append(r) # now add the rule to the style and we're done

m.append_style('My Style',s) # Styles are given names only as they are applied to the map

ds = mapnik.Shapefile(file='test.shp')

ds.envelope()

layer = mapnik.Layer('pz_koeln') # new layer called 'world' (we could name it anything)
# note: layer.srs will default to '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

layer.datasource = ds

layer.styles.append('My Style')

m.layers.append(layer)
m.zoom_all()

# Write the data to a png image called world.png in the current directory
mapnik.render_to_file(m,'paketzentrum.png', 'png')

# Exit the Python interpreter
exit() # or ctrl-d
