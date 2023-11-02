

 * sudo apt-get -y install pdal

 begin trasnforming to usefule format:

 * pdal translate Ruddington_ResStr.laz Ruddington_ResStr.pcd


 laz -> csv -> pcd color:

  * pdal pipeline pipeline.json
  * python3 extract_pcd_from_csv_with_color.py Ruddington_ResStr.csv


do other things:


 * python3 downsample Ruddington_ResStr.pcd

 * python3 pcdview downsample_Ruddington_ResStr.pcd

