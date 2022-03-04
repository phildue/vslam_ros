#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import sys
import math
_EPS = np.finfo(float).eps * 4.0

def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array([
        [                1.0,                 0.0,                 0.0, t[0]]
        [                0.0,                 1.0,                 0.0, t[1]]
        [                0.0,                 0.0,                 1.0, t[2]]
        [                0.0,                 0.0,                 0.0, 1.0]
        ], dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]],
        [    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]],
        [    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]],
        [                0.0,                 0.0,                 0.0, 1.0]
        ], dtype=np.float64)

def read_trajectory(filename, matrix=True):
    """
    Read a trajectory from a text file. 
    
    Input:
    filename -- file to be read
    matrix -- convert poses to 4x4 matrices
    
    Output:
    dictionary of stamped 3D poses
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[float(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list_ok = []
    for i,l in enumerate(list):
        if l[4:8]==[0,0,0,0]:
            continue
        isnan = False
        for v in l:
            if np.isnan(v): 
                isnan = True
                break
        if isnan:
            sys.stderr.write("Warning: line %d of file '%s' has NaNs, skipping line\n"%(i,filename))
            continue
        list_ok.append(l)
    if matrix :
      traj = dict([(l[0],transform44(l[0:])) for l in list_ok])
    else:
      traj = dict([(l[0],l[1:8]) for l in list_ok])
    return traj
import math
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def isRotationMatrix(R) :
    """
     Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R) :
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def ominus(a,b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(a),b)

if __name__ == "__main__":
        traj = read_trajectory("/media/data/dataset/rgbd_dataset_freiburg1_xyz/groundtruth.txt")

        t = np.array([np.array(t) for t in traj.keys()],dtype=float)
        poses = np.array([np.array(v) for v in traj.values()],dtype=float)
        velocities = np.zeros((poses.shape[0],6))

        for i in range(1,poses.shape[0]):
                relative_motion = ominus(poses[i-1],poses[i])
                dt = t[i] - t[i-1]
                eulers = rotationMatrixToEulerAngles(relative_motion[:3,:3])
                velocities[i,:3] = relative_motion[:3,3]/dt
                velocities[i,3:] = np.array(eulers)/dt #?
                

        mean = velocities.mean(axis=0)
        std = velocities.std(axis=0)
        velocities = (velocities - mean)

        DT = 0.1
        relative_motion = velocities * DT

        cov = np.cov(relative_motion.transpose())

        print ( "Mean: \n{}".format(np.round(relative_motion.mean(axis=0),4)) )
        print ( "Cov: \n{}".format(np.round(cov,4) ))

        plt.figure()
        for i,t in enumerate(["tx","ty","tz"]):
                plt.subplot(2,3,i+1)
                plt.title(t)
                plt.hist(relative_motion[:,i])
                plt.xlabel("Relative Translational Motion [m] ")
        for i,t in enumerate(["ax","ay","az"],3):
                plt.subplot(2,3,i+1)
                plt.title(t)
                plt.hist(relative_motion[:,i]/math.pi*180.0)
                plt.xlabel("Relative Angular [°] ")
        plt.show()
