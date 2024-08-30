# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:57:46 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import PIL
from scipy import signal as sig
from scipy import signal
from scipy import ndimage


import matplotlib.colors

#%matplotlib qt5

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d/np.sum(gkern2d)

def geometricDistance(point1,point2):
    diff = point1-point2
    return np.sqrt(diff[0]**2+diff[1]**2)
        
def smoothpoints(Vpoints,scale,iterations):
    for i in range(iterations):
        print('Cycle: {}'.format(i+1))
        for point1 in Vpoints:
            nudge = np.zeros(2)
            for point2 in Vpoints:
                if point1[0] != point2[0] and point1[1] != point2[1]:
                    distance = geometricDistance(point1, point2)
                    if distance < 0.1:
                        diff = point1-point2
                        direction = diff/distance
                        nudge += direction * scale
                    npoint = point1 + nudge
                    if npoint[0] > 1 or npoint[0] < 0 or npoint[1] > 1 or npoint[1] < 0:
                        nudge = [0,0]
            point1 += nudge
    return Vpoints

def vorEdgeSmooth(vor):
    for vertex in vor.vertices:
        if vertex[0] < 0 and vertex[1] < 0:
            vertex = np.array([0,0])
        if vertex[0] < 1 and vertex[1] < 1:
            vertex = np.array([1,1])
        if vertex[0] < 0 and vertex[1] < 1:
            vertex = np.array([0,1])
            if vertex[0] < 1 and vertex[1] < 0:
                vertex = np.array([1,0])
    return vor
        
def collapseOcean(vor,p,cycles):
    oceanPolygons = []
    oceanPointRegions = []
    for point_region in vor.point_region:
        region = vor.regions[point_region]
        if not -1 in region:
            polygon = []
            ocean = False
            for i in region:
                v = vor.vertices[i]
                if v[0] <= 15/p or v[1] <= 15/p or v[0] >= 1 - 15/p or v[1] >= 1 - 15/p:
                    ocean = True
                polygon.append(v)
            if ocean == True:    
                oceanPolygons.append(polygon)
                oceanPointRegions.append(point_region)
    for c in range(cycles):
        for point_region in vor.point_region:
            region = vor.regions[point_region]
            if not -1 in region:
                polygon = []
                ocean = False
                for i in region:
                    v = vor.vertices[i]
                    if any(v[0] == point[0] for point in [x for xs in oceanPolygons for x in xs]) or any(v[1] == point[1] for point in [x for xs in oceanPolygons for x in xs]):
                        if np.random.rand() > 0.9:
                            ocean = True
                    polygon.append(v)
                if ocean == True:    
                    oceanPolygons.append(polygon)
                    oceanPointRegions.append(point_region)
    return oceanPolygons, oceanPointRegions    

def rasterize(vor, oceanPointRegions, fig_size = 5.12):
    land = [l for l in vor.point_region if l not in oceanPointRegions]
    polygons = []
    for l in land:
        region = vor.regions[l]
        if not -1 in region:
            polygon = []
            ocean = False
            for i in region:
                v = vor.vertices[i]
                polygon.append(v)
            polygons.append(polygon)
            
    fig = voronoi_plot_2d(vor,show_vertices=False,  show_points = False, line_width = 0)
    fig.patch.set_facecolor('white')
    fig.set_figwidth(fig_size)
    fig.set_figheight(fig_size)
    for poly in polygons:
        plt.fill(*zip(*poly),color='black')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.axis('off')
    fig.canvas.draw()
    temp_canvas = fig.canvas
    plt.close()
    
    
    pil_image = PIL.Image.frombytes('RGB', temp_canvas.get_width_height(),  temp_canvas.tostring_rgb())
    rImage = np.asarray(pil_image)[:,:,0]
    return rImage

def makeMountains(rImage):
    '''
    Needs generalising for many input sizes i.e. number of convolutions and kernel size
    
    Parameters
    ----------
    rImage : TYPE
        DESCRIPTION.

    Returns
    -------
    mountains : TYPE
        DESCRIPTION.

    '''
    mountains = rImage
    for i in range(24):
        #rImage = sig.convolve2d(rImage,gkern(6),mode='same',boundary ='wrap')
        mountains = sig.convolve2d(mountains,gkern(8),mode='same',boundary ='wrap')
    mountains = mountains/np.max(mountains)
    mountains+=1e-32
    mountains = -np.log2(mountains)
    
    return mountains


def chooseLocations(terrain, bounds, lakes = 3):
    raffle = np.where(np.logical_and(terrain < bounds[1], terrain > bounds[0]))
    rPos = []
    for i in range(lakes):
        i = np.random.randint(raffle[0].shape[0])
        rPos.append([raffle[0][i],raffle[1][i]])
    lakeLocals = np.asarray(rPos)
    return lakeLocals

def createLake(lake_depth):
    pl = 64
    vPointsl = np.random.rand(pl,2)

    vorl = Voronoi(vPointsl)
    smoothl = vPointsl


    vorl = Voronoi(smoothl)
    vorl = vorEdgeSmooth(vorl)
    oceanPolygonsl, oceanPointRegionsl = collapseOcean(vorl,pl,2)
    rImagel = rasterize(vorl, oceanPointRegionsl,1)
    lake = np.copy(rImagel)/255

    lake[lake==0] = lake_depth
    lake[lake>lake_depth] = 0
    return lake

def plotRiverPath(peaks,lakes,riverMouths):
    rLakes = [np.asarray([0,0])]
    for peak in peaks:
        lakeStep = np.asarray([0,0])
        minDis = 512
        for lake in lakes:
            dis = geometricDistance(peak,lake)
            if dis < minDis:# and [rLakes != lake][0].any()):
                if [rLakes != lake][0].all():
                    minDis = dis
                    lakeStep = lake
        rLakes.append(lakeStep)
    rLakes = np.asarray(rLakes)[1:]
    rMouths = [np.asarray([0,0])]
    for lake in rLakes:
        mouthStep = np.asarray([0,0])
        minDis = 512
        for mouth in riverMouths:
            dis = geometricDistance(lake,mouth)
            if dis < minDis:# and [rLakes != lake][0].any()):
                if [rMouths != mouth][0].all():
                    minDis = dis
                    mouthStep = mouth
        rMouths.append(mouthStep)
    rMouths = np.asarray(rMouths)[1:]
    return peaks,rLakes,rMouths

# def placeRiverMouths(terrain, rMouths):
#     grad = np.gradient(terrain)
#     # generate delta directions.
#     mDir = []
#     mRange = 5
#     for m in rMouths: 
#         Dir = np.mean(grad[0][m[0]-mRange:m[0]+mRange, m[1]-mRange:m[1]+mRange]), np.mean(grad[1][m[0]-mRange:m[0]+mRange, m[1]-mRange:m[1]+mRange])
#         Dir = Dir/np.linalg.norm(Dir)
#         mAngle = np.arctan2(Dir[1],Dir[0])
#         mDir.append(Dir)
#         mSize = 21
#         mShape = np.zeros((mSize,mSize))
#         half = int(mSize/2)
#         mShape[:,half] = 1
#         for i in range(mSize):
#             mShape[mSize-i-1,int(half-i/3):-int(half-i/3)] = 1
#         mShape = ndimage.rotate(mShape,np.rad2deg(mAngle))
#         rHalf = int(mShape.shape[0]/2)
#         mShape/=np.max(mShape)
#         mShape*= 0.1
#         sDir = Dir*5
#         if (mShape.shape[0] % 2) == 0:   
#             terrain[int(m[0]+sDir[0])-rHalf:int(m[0]+sDir[0])+rHalf,int(m[1]+sDir[1])-rHalf:int(m[1]+sDir[1])+rHalf][mShape>0.01] = mShape[mShape>0.01]
#         else:
#             terrain[int(m[0]+sDir[0])-rHalf:int(m[0]+sDir[0])+rHalf+1,int(m[1]+sDir[1])-rHalf:int(m[1]+sDir[1])+rHalf+1][mShape>0.01] = mShape[mShape>0.01]
#     mDir = np.asarray(mDir)  
#     return terrain



def littleMountains(terrain, island, number):
    mountain = gkern(np.random.randint(12,24)*2)
    mountain /= (np.max(mountain)/1.4)
    loc = chooseLocations(terrain,(0.65,0.85),number)
    sh = mountain.shape[0]
    for l in loc:
        terrain[l[0]-int(sh/2):l[0]+int(sh/2),l[1]-int(sh/2):l[1]+int(sh/2)][mountain>0.6] = mountain[mountain>0.6]
    return terrain

def createIsland():
    pl = 64
    vPointsl = np.random.rand(pl,2)

    vorl = Voronoi(vPointsl)
    smoothl = vPointsl


    vorl = Voronoi(smoothl)
    vorl = vorEdgeSmooth(vorl)
    oceanPolygonsl, oceanPointRegionsl = collapseOcean(vorl,pl,2)
    rImagel = rasterize(vorl, oceanPointRegionsl,1)
    island = -((np.copy(rImagel)/255) - 0.5)
    island[island<0.1] = 0.01
    return island

def placeRiverMouth(terrain, m):
    grad = np.gradient(terrain)
    # generate delta directions.
    mRange = 5
    Dir = np.mean(grad[0][m[0]-mRange:m[0]+mRange, m[1]-mRange:m[1]+mRange]), np.mean(grad[1][m[0]-mRange:m[0]+mRange, m[1]-mRange:m[1]+mRange])
    Dir = Dir/np.linalg.norm(Dir)
    mAngle = np.arctan2(Dir[1],Dir[0])
    mSize = 21
    mShape = np.zeros((mSize,mSize))
    half = int(mSize/2)
    mShape[:,half] = 1
    for i in range(mSize):
        mShape[mSize-i-1,int(half-i/3):-int(half-i/3)] = 1
    mShape = ndimage.rotate(mShape,np.rad2deg(mAngle))
    rHalf = int(mShape.shape[0]/2)
    mShape/=np.max(mShape)
    mShape*= 0.1
    sDir = Dir*5
    if (mShape.shape[0] % 2) == 0:   
        terrain[int(m[0]+sDir[0])-rHalf:int(m[0]+sDir[0])+rHalf,int(m[1]+sDir[1])-rHalf:int(m[1]+sDir[1])+rHalf][mShape>0.01] = mShape[mShape>0.01]
    else:
        terrain[int(m[0]+sDir[0])-rHalf:int(m[0]+sDir[0])+rHalf+1,int(m[1]+sDir[1])-rHalf:int(m[1]+sDir[1])+rHalf+1][mShape>0.01] = mShape[mShape>0.01]  
    return terrain

def placeRiverSegment(terrain, m, Dir):
    mAngle = np.arctan2(Dir[1],Dir[0])
    mSize = 10
    mShape = np.zeros((mSize,mSize))
    mShape[:,3:-3] = 1
        
    mShape = ndimage.rotate(mShape,np.rad2deg(mAngle))
    rHalf = int(mShape.shape[0]/2)
    mShape/=np.max(mShape)
    mShape*= 0.1
    sDir = Dir*5
    if (mShape.shape[0] % 2) == 0:   
        terrain[int(m[0]+sDir[0])-rHalf:int(m[0]+sDir[0])+rHalf,int(m[1]+sDir[1])-rHalf:int(m[1]+sDir[1])+rHalf][mShape>0.01] = mShape[mShape>0.01]
    else:
        terrain[int(m[0]+sDir[0])-rHalf:int(m[0]+sDir[0])+rHalf+1,int(m[1]+sDir[1])-rHalf:int(m[1]+sDir[1])+rHalf+1][mShape>0.01] = mShape[mShape>0.01]  
    return terrain

def riverDescent(terrain,m):
    grad = np.gradient(terrain)
    newTerrain = np.copy(terrain)
    rivers = []
    mRange = 2
    ss = 5
    pos = np.copy(m)
    river = [pos]
    #riverlength
    for i in range(50):
        Dir = np.mean(grad[0][pos[0]-mRange:pos[0]+mRange, pos[1]-mRange:pos[1]+mRange]), np.mean(grad[1][pos[0]-mRange:pos[0]+mRange, pos[1]-mRange:pos[1]+mRange])
        Dir = Dir/np.linalg.norm(Dir)
        pos[0] = pos[0] - ss*Dir[0]
        pos[1] = pos[1] - ss*Dir[1]
        newTerrain = placeRiverSegment(newTerrain, pos, Dir)
        if terrain[pos[0],pos[1]]<0.13:
            newTerrain = placeRiverMouth(newTerrain, (pos[0],pos[1]))
            break
        river.append([pos[0],pos[1]])
    return newTerrain, np.asarray(river)    
    
        

# Generating Vononoi Polygons.
p = 225
vPoints = np.random.rand(p,2)

# poin

vor = Voronoi(vPoints)

print('Smoothing Land...')
smooth = smoothpoints(vPoints,0.01,1)


vor = Voronoi(smooth)
vor = vorEdgeSmooth(vor)
print('Collapsing Oceans...')
oceanPolygons, oceanPointRegions = collapseOcean(vor,p,3)
rImage = rasterize(vor, oceanPointRegions)

# Geographic processing
mountains = makeMountains(rImage)

island = 1-(rImage/255)
#smooth island
for i in range(4):
    island = sig.convolve2d(island,gkern(8),mode='same',boundary ='wrap')   
    
terrain = (4*island # regular area
           +(1/16)*mountains)

terrain /= np.max(terrain)



lakes = chooseLocations(terrain, (0.3,0.7), 16)
lake_depth = 0.05


# print('Plotting Lakes...')
# for l in range(lakes.shape[0]):
#     lx, ly = lakes[l]
#     lake = createLake(lake_depth)
#     if lx-50>0 and ly-50>0 and lx+50<512 and ly+50<512:
#         terrain[lx-50:lx+50,ly-50:ly+50][lake==lake_depth] = lake[lake==lake_depth]


print('Growing Lonely Mountain...')
terrain = littleMountains(terrain, island, 1)


print('Scattering Little Islands...')
islands = chooseLocations(terrain, (1e-10,1e-4), 6)

for i in range(islands.shape[0]):
    lx, ly = islands[i]
    for j in range(4):
        island = sig.convolve2d(createIsland(),gkern(16), mode='same')
    if lx-50>0 and ly-50>0 and lx+50<512 and ly+50<512:
        terrain[lx-50:lx+50,ly-50:ly+50][island>0.2] = island[island>0.2]


# Place towns
towns = chooseLocations(terrain, (0.2,0.25), 10)

#smooth terrain
terrain = sig.convolve2d(terrain,gkern(6),mode='same',boundary ='wrap') 

#add noise
terrain+=(1/16)*np.random.rand(*rImage.shape)

print('Carving Rivers...')
rMountains = chooseLocations(terrain, (0.85,0.95), 6)
for rMountain in rMountains:
    terrain, river = riverDescent(terrain,rMountain)

print('Plotting Lakes...')
for l in range(lakes.shape[0]):
    lx, ly = lakes[l]
    lake = createLake(lake_depth)
    if lx-50>0 and ly-50>0 and lx+50<512 and ly+50<512:
        terrain[lx-50:lx+50,ly-50:ly+50][lake==lake_depth] = lake[lake==lake_depth]


#IMPORTANT\/\/\/\/\/\/
#another set of smoothing and noise
terrain = sig.convolve2d(terrain,gkern(6),mode='same',boundary ='wrap') 
terrain+=(1/16)*np.random.rand(*rImage.shape)



deepOcean = ["#41A5B4"]
shallowOcean = ["#96CDD2"]
sand = ["#E1D791"]
darkGrass = ["#46AF64"]
grass = ["#70c048"]

lightGrass = ["#96C869"]
#lighterGrass = ["#c0f098"]
rock = ["#CDB991"]
snow = ["#FFFFFF"]



cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 2*deepOcean+shallowOcean+sand+2*darkGrass+4*grass+4*lightGrass+2*rock+2*snow)

plt.imshow(terrain,
          cmap=cmap)
#plt.scatter(towns[:,1],towns[:,0],color='r')
#plt.scatter(lakes[:,1],lakes[:,0],color='cyan')
#plt.scatter(peaks[:,1],peaks[:,0],color='brown')
#plt.scatter(rLakes[:,1],rLakes[:,0],color='r')
#for i in range(rMouths.shape[0]):
#     plt.arrow(rMouths[i,1],rMouths[i,0],10*mDir[i,1],10*mDir[i,0],color='blue')
# for i in range(len(rivers)):
#     plt.scatter(rivers[i][:,1],rivers[i][:,0])
#plt.scatter(river[:,1],river[:,0])
plt.axis('off')
plt.show()
