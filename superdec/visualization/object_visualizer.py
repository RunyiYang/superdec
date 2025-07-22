import numpy as np
import os
import viser
import time
from superdec.utils.predictions_handler import PredictionHandler 
from superdec.utils.visualizations import generate_ncolors

RESOLUTION = 25

def main():
  server = viser.ViserServer()
  
  epoch = 2000
  dataset = 'scene'
  split = 'room2'
  name_experiment = '18-07-multigpu-normalized-low-sps-2nd-aprt'
  base_path = '/ephemeral/outputs'
  
  input_path = os.path.join(base_path, name_experiment, f'{dataset}_{str(epoch)}_{split}.npz')

  print("Opening npz...")
  predictions_sq = PredictionHandler.from_npz(input_path)
  print("Computing meshes...")
  meshes = predictions_sq.get_meshes(resolution=RESOLUTION)
  print("Computing point clouds...")
  pcs = predictions_sq.get_segmented_pcs()
  print("Done!")
  names = predictions_sq.names

  if dataset == 'scene': # visualization for scenes
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    colors = generate_ncolors(len(meshes))/255.0 # Color each object with a different color
    for idx in range(len(meshes)):
      if meshes[idx] == None or int(names[idx]) > 200: # TODO for now I am only taking the first 200 objects, modify this
        continue
      meshes[idx].visual.face_colors = np.ones((meshes[idx].visual.face_colors.shape[0], 3)) * colors[idx]
      meshes[idx].visual.vertex_colors = np.ones((meshes[idx].visual.vertex_colors.shape[0], 3)) * colors[idx]
      server.scene.add_mesh_trimesh(f"superquadrics_{names[idx]}", mesh=meshes[idx], visible=True)
    
      server.scene.add_point_cloud(
            name=f"/segmented_pointcloud_{names[idx]}",
            points=np.array(pcs[idx].points),
            colors=pcs[idx].colors,
            point_size=0.005,
            visible = True
        )
  else: # visualization for objects
    server.scene.set_up_direction([0.0, 1.0, 0.0])
    gui_model_selection = server.gui.add_dropdown("Model index", [str(i) for i in range(len(names))], initial_value='0')
    gui_model_selection.on_update(lambda _: draw_superquadric_and_segmentation())
    draw_superquadric_and_segmentation()

  def draw_superquadric_and_segmentation():
    idx = int(gui_model_selection.value)
    server.scene.add_mesh_trimesh("superquadrics", mesh=meshes[idx], visible=True)
    
    server.scene.add_point_cloud(
          name="/segmented_pointcloud",
          points=np.array(pcs[idx].points),
          colors=np.array(pcs[idx].colors),
          point_size=0.005,
      )
    
  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (0.8, 0.8, 0.8)
    client.camera.look_at = (0., 0., 0.)
    
  
  while True:
      time.sleep(10.0)

if __name__ == '__main__':
  main()