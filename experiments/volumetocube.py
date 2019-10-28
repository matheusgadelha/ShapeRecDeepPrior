import numpy as np
import sys
import glob
import binvox_rw
import os

def write_cubes_obj(path, points, faces):
    f = open(path, 'w')
    for p in points:
      f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
    for q in faces:
      f.write("f {} {} {} {}\n".format(q[0], q[1], q[2], q[3]))


def volume_to_cubes(volume, threshold=0, dim=[2., 2., 2.]):
    o = np.array([-dim[0]/2., -dim[1]/2., -dim[2]/2.])
    step = np.array([dim[0]/volume.shape[0], dim[1]/volume.shape[1], dim[2]/volume.shape[2]])
    points = []
    faces = []
    for x in range(1, volume.shape[0]-1):
        for y in range(1, volume.shape[1]-1):
            for z in range(1, volume.shape[2]-1):
                pos = o + np.array([x, y, z]) * step
                if volume[x, y, z] > threshold:
                    vidx = len(points)+1
                    POS = pos + step*0.95
                    xx = pos[0]
                    yy = pos[1]
                    zz = pos[2]
                    XX = POS[0]
                    YY = POS[1]
                    ZZ = POS[2]
                    points.append(np.array([xx, yy, zz]))
                    points.append(np.array([xx, YY, zz]))
                    points.append(np.array([XX, YY, zz]))
                    points.append(np.array([XX, yy, zz]))
                    points.append(np.array([xx, yy, ZZ]))
                    points.append(np.array([xx, YY, ZZ]))
                    points.append(np.array([XX, YY, ZZ]))
                    points.append(np.array([XX, yy, ZZ]))
                    faces.append(np.array([vidx, vidx+1, vidx+2, vidx+3]))
                    faces.append(np.array([vidx, vidx+4, vidx+5, vidx+1]))
                    faces.append(np.array([vidx, vidx+3, vidx+7, vidx+4]))
                    faces.append(np.array([vidx+6, vidx+2, vidx+1, vidx+5]))
                    faces.append(np.array([vidx+6, vidx+5, vidx+4, vidx+7]))
                    faces.append(np.array([vidx+6, vidx+7, vidx+3, vidx+2]))
    return points, faces


def write_obj_from_array(path, volume):
    print("Saving {}...".format(path))
    pts, faces = volume_to_cubes(volume, threshold=0.5)
    write_cubes_obj(path, pts, faces)
    print("Done.")


def write_bin_from_array(path, volume):
    print("Saving {}...".format(path))
    binary = (volume > 0.3).astype('int8')
    np.save(path, binary)
    print("Done.")


if __name__ == '__main__':
    volumefile = sys.argv[1]
    outfile = sys.argv[2]
    volume = np.load(sys.argv[1])
    volume = volume.astype('float32')

    pts, faces = volume_to_cubes(volume, threshold=0.5)
    write_cubes_obj(outfile, pts, faces)
#    os.system('meshlab .out.obj')
# os.system('rm .out.obj')


