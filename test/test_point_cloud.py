import open3d as o3d

import cv2
import numpy as np

from matplotlib.colors import Normalize
from matplotlib.colors import SymLogNorm
from matplotlib import cm

import matplotlib.pyplot as plt

from test_dlis_load_function import dlis_loader

def main ():
    file = '.\\data\\58-32_processed_image\\DLIS_XML_ProcessedImages\\University_of_Utah_MU-ESW1_FMI-HD_7390-7527ft_Run3.dlis'
    #file = '.\\data\\58-32_FMI_DLIS_XML\\University_of_Utah_MU_ESW1_FMI_HD_2226_7550ft_Run1.dlis'
    #file = '.\\data\\58-32_FMI_DLIS_XML\\University_of_Utah_MU_ESW1_FMI_HD_7440_7550ft_Run2.dlis'

    curves_frame = dlis_loader(file)
    tdep = curves_frame['TDEP']
    P1NO_FBST_S = curves_frame['P1NO_FBST_S']
    RB_FBST_S = curves_frame['RB_FBST_S']
    HAZIM_S = curves_frame['HAZIM_S']
    DEVIM_S = curves_frame['DEVIM_S']
    C1_S = curves_frame['C1_S']
    C2_S = curves_frame['C2_S']
    BS = curves_frame['BS']
    fmi_dyn = curves_frame['FMI_DYN']
    fmi_dyn[fmi_dyn==-9999]=np.nan
    fmi_stat = curves_frame['FMI_STAT']
    fmi_stat[fmi_stat==-9999]=np.nan

    # for k, fmi in {'stat':fmi_stat, 'dyn':fmi_dyn}.items():
    #     if k=='stat': norm = SymLogNorm(linthresh=0.1, linscale=1.0, vmin=np.nanmin(fmi) if np.nanmin(fmi)>0.0 else 0.0, vmax=np.nanmax(fmi), clip=False, base=10)
    #     else: norm = Normalize(vmin=np.nanmin(fmi) if np.nanmin(fmi)>0.0 else 0.0, vmax=np.nanmax(fmi), clip=False)
    #     mapper = cm.ScalarMappable(norm=norm, cmap=cm.YlOrBr)
    #     fmi_raster = mapper.to_rgba(fmi, bytes=True)
    #     cv2.imwrite(f'fmi_{k}.png', cv2.cvtColor(fmi_raster, cv2.COLOR_RGBA2BGRA))
    #     cv2.imshow("image", cv2.cvtColor(fmi_raster, cv2.COLOR_RGBA2BGRA))
    #     cv2.waitKey()
    
    
    #norm = Normalize(vmin=np.nanmin(fmi_dyn) if np.nanmin(fmi_dyn)>0.0 else 0.0, vmax=np.nanmax(fmi_dyn), clip=False)
    norm = SymLogNorm(linthresh=0.1, linscale=1.0, vmin=np.nanmin(fmi_dyn) if np.nanmin(fmi_dyn)>0.0 else 0.0, vmax=np.nanmax(fmi_dyn), clip=False, base=10)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.YlOrBr)
    fmi_raster = mapper.to_rgba(fmi_dyn, bytes=False,)

    fmi_raster = np.array([i for i in fmi_raster[15100:-1,:,:]])

    steps = [0]
    for _ in range(199):
        steps.append(steps[_]+np.sin(1))

    #fmi_xyz = np.array([[[4.25*np.cos(i*0.0174532925), 4.25*np.sin(i*0.0174532925), z/12] for i in range(fmi_raster.shape[1])] for z in range(fmi_raster.shape[0])])
    fmi_xyz = np.array([[[np.cos(i*0.0174532925)*(4.25+(fmi_raster[z,i,0]/0.1)), np.sin(i*0.0174532925)*(4.25+(fmi_raster[z,i,0]/0.1)), z/10] for i in range(fmi_raster.shape[1])] for z in range(fmi_raster.shape[0])])
    #fmi_xyz = np.array([[[0.0254*np.cos(i)*100, 0.0254*np.sin(i)*100, z] for i in range(fmi_raster.shape[1])] for z in range(fmi_raster.shape[0]])
    #fmi_xyz = np.array([[[i, 0, z] for i in range(fmi_raster.shape[1])] for z in range(fmi_raster.shape[0])])

    fmi_xyz_n = np.array([[[1*np.cos(i*0.0174532925)/(4.25*2), 1*np.sin(i*0.0174532925)/(4.25*2), 0.0] for i in range(fmi_raster.shape[1])] for z in range(fmi_raster.shape[0])])
    #fmi_xyz_n = fmi_xyz_n / np.linalg.norm(fmi_xyz_n)

    fmi_raster = fmi_raster.reshape(-1, fmi_raster.shape[-1])
    fmi_xyz = fmi_xyz.reshape(-1, fmi_xyz.shape[-1])
    fmi_xyz_n = fmi_xyz_n.reshape(-1, fmi_xyz_n.shape[-1])

    todrop = [row for row in range(fmi_raster.shape[0]) if fmi_raster[row].all()==0]
            
    fmi_raster = np.delete(fmi_raster, todrop, axis=0)
    fmi_xyz = np.delete(fmi_xyz, todrop, axis=0)
    fmi_xyz_n = np.delete(fmi_xyz_n, todrop, axis=0)

    #np.save('fmi_raster', fmi_raster)
    #np.save('fmi_xyz', fmi_xyz)
    #np.save('fmi_xyz_n.npy', fmi_xyz_n)



    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fmi_xyz)
    pcd.colors = o3d.utility.Vector3dVector(fmi_raster[:,:3])
    pcd.normals = o3d.utility.Vector3dVector(fmi_xyz_n)
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8.5, max_nn=36))
    #o3d.geometry.orient_normals_to_align_with_direction(pcd, orientation_reference=[0., 0., 0.])
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    o3d.io.write_point_cloud("pcd.ply", pcd, write_ascii=True)


    #mesh = o3dtut.get_bunny_mesh()
    #pcd = mesh.sample_points_poisson_disk(750)
    #pcd = o3d.io.read_point_cloud("pcd.ply")

    # # estimate radius for rolling ball
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist   

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #         pcd,
    #         o3d.utility.DoubleVector([radius, radius * 2]))
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # alpha = 0.5
    # print(f"alpha={alpha:.3f}")
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
    #     print(f"alpha={alpha:.3f}")
    #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    #     mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    print('run Poisson surface reconstruction')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=13, width=0, scale=1, linear_fit=True)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    print('visualize densities')
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    o3d.visualization.draw_geometries([density_mesh], mesh_show_back_face=True)

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    #mesh = mesh.simplify_quadric_decimation(100)
    print(mesh)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("mesh.ply", mesh, write_ascii=True)

    
    
    print ('hello')

def new_func():
    return 15

if __name__ == "__main__":
    main()