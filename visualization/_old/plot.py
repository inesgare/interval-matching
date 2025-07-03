import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['savefig.facecolor']='white' # set background to white in savefig
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

### PLOT POINT CLOUDS

# plot point clouds in 2D and 3D

def plot_point_cloud(pts, zoom_factor = 1) :
    if pts.shape[1] == 2 : # 2D
        xx, yy = pts.T
        plt.figure(figsize = plt.figaspect(1.)*zoom_factor)
        plt.scatter(xx, yy, c = '#729dcf', s = 50, edgecolors='black')
        plt.axis('equal')
        plt.show()
    if pts.shape[1] == 3 : # 3D
        xx,yy,zz = pts.T
        fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xx, yy, zz, c = '#729dcf', s = 20, edgecolors='black')
        plt.show()
        
def plot_two_point_clouds(X,Y, zoom_factor = 1) :
    if X.shape[1] == 2 : # 2D
        plt.figure(figsize=plt.figaspect(1.)*zoom_factor)
        plt.scatter(X[:,0], X[:,1], c = 'r', marker = 'x')
        plt.scatter(Y[:,0], Y[:,1], c = 'k', marker = 'o', alpha = .2)
        plt.axis('equal')
        plt.show()
    if X.shape[1] == 3 : # 3D
        fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:,0],X[:,1],X[:,2], c = 'r', marker = 'x', s = 20)
        ax.scatter(Y[:,0],Y[:,1],Y[:,2], c = 'k', marker = 'o', s = 20)
        plt.show()
    
### PLOT DIAGRAMS
        
# Functions to plot persistence diagrams

def plot_diagrams(diagrams, style = 'sep', diagonal = False, show_inf = False, thr_inf = None) :
    if type(diagrams) == np.ndarray :
        PH_list = list(get_PH_alldims(diagrams))
    else :
        PH_list = [ np.array(diag) for diag in diagrams ]
    
    ph0 = len(PH_list[0])
    ph1 = len(PH_list[1])
    ph2 = len(PH_list[2])   
    
    PH0 = PH_list[0] # in-place modif
    if show_inf :
        PH0[PH0 == np.inf] = thr_inf 
    else :
        PH0[PH0 == np.inf] = 0 # reduce to trivial bar
    
    col_list = ['#E11033','#1A0DAB','#FF7900']# FBB117
         
    if ph2 != 0 :
        list_dims = [0,1,2]
        figsize = (10, 3.5)
    elif ph1 != 0 :
        list_dims = [0,1]
        figsize = (6.5,3.5)
    else :
        list_dims = [0]
        figsize = (6.5,3.5)
    
    if diagonal == False :
        for dim in list_dims :
            PHdim = PH_list[dim]
            if len(PHdim > 0) :
                PH_list[dim] = PHdim[ PHdim[:,0] < PHdim[:,1] ]
            
    if style == 'sep' :
        fig, axes = plt.subplots(1,len(list_dims), figsize = figsize)
        for dim in list_dims :
            if len(list_dims) > 1 :
                ax = axes[dim]
            else :
                ax = axes
            if len(PH_list[dim]) > 0 :
                xmin = PH_list[dim][:,0].min()
                ymax = PH_list[dim][:,1].max()
                ax.plot([xmin, ymax],[xmin, ymax], c = 'k', alpha = .2, zorder = 1)
                ax.scatter(PH_list[dim][:,0], PH_list[dim][:,1], c = col_list[dim], s = 5, zorder = 2)
                ax.set_aspect('equal')
                ax.set_title('PH{}'.format(dim))
                ax.set_xlabel('birth')
                ax.set_ylabel('death')
        fig.tight_layout()
        plt.show()
        
    if style == 'tog' :
        fig = plt.figure(figsize = (4.5,4.5))
        xmin = min( [ PH_list[dim][:,0].min() for dim in [0,1,2] ] )
        ymax = max( [ PH_list[dim][:,1].max() for dim in [0,1,2] ] )
        plt.plot([xmin, ymax],[xmin, ymax], c = 'k',alpha = .2, label='_nolegend_')
        for dim in list_dims :
            plt.scatter(PH_list[dim][:,0], PH_list[dim][:,1], c = col_list[dim], s = 10)
        plt.legend(['PH0','PH1','PH2'])
        fig.tight_layout()
        plt.show()
        
from matplotlib.lines import Line2D

def plot_bars(diagrams, diagonal = False, show_inf = False, thr_inf = None,
              delta_y = .1, big_delta_y = .15) :
    
    if type(diagrams) == np.ndarray :
        PH_list = list(get_PH_alldims(diagrams))
    else :
        PH_list = [ np.array(diag) for diag in diagrams ]
        
    ph0 = len(PH_list[0])
    ph1 = len(PH_list[1])
    ph2 = len(PH_list[2])        
    
    PH0 = PH_list[0] # in-place modif
    if show_inf :
        PH0[PH0 == np.inf] = thr_inf # np.max(PH0[PH0 < np.inf]) + thr_inf
    else :
        PH0[PH0 == np.inf] = 0 # reduce to trivial bar
    
    if ph2 != 0 :
        list_dims = [0,1,2]
    elif ph1 != 0 :
        list_dims = [0,1]
    else :
        list_dims = [0]
        
    if diagonal == False :
        for dim in list_dims :
            PHdim = PH_list[dim]
            if len(PHdim > 0) :
                PH_list[dim] = PHdim[ PHdim[:,0] < PHdim[:,1] ]
    
    col_list = ['#E11033','#1A0DAB','#FF7900']# FBB117
    lab_list = ['PH0','PH1','PH2']
    xmax = max([PH_list[dim][:,1].max() for dim in list_dims if len(PH_list[dim]) > 0 ] )
    ymax = delta_y*(ph0 + ph1 + ph2) + 2*big_delta_y
    
    fig = plt.figure(figsize = (6, 5))
    y = 0 # increment the position of the bar
    for dim in list_dims :
        for bar in PH_list[dim] :
            birth, death = bar
            plt.plot([birth, death],[y, y], c = col_list[dim], linewidth = 1)
            y += delta_y
        y += big_delta_y
    plt.axis('equal')
    plt.yticks([])
    plt.xlim([-delta_y,xmax+delta_y])
    plt.ylim([-delta_y,ymax+delta_y])
    plt.ylabel('persistence bars')
    plt.xlabel('filtration value')
    fig.tight_layout()
    legend_elements = [ Line2D([0], [0], color=col_list[dim], lw=1, label=lab_list[dim]) for dim in list_dims]
    plt.legend(handles=legend_elements)
    plt.show()
    
### PLOT REPS

# plot representatives in 2D / 3D

# 2D with matplotlib

def plot_cycreps_2D(pts, list_cycles, pts_to_show = None, ax = None, return_ax = False):
    ''' cycreps is a list of 1D cycles, i.e.
    a list of list of couples giving the 2 indices to take from pts (endpoints)'''
    
    xx = pts[:,0]
    yy = pts[:,1]
    
    ax_was_None = False
    if ax is None :
        ax_was_None = True
        plt.figure()
        ax = plt.gca()
    for cycle in list_cycles :
        if len(cycle) > 0 :
            cycle = np.array(cycle)
            ax.plot(xx[cycle.T], yy[cycle.T], c = 'k', alpha = 0.5)
    if pts_to_show is None :
        pts_to_show = pts.copy()
    ax.scatter(pts_to_show[:,0], pts_to_show[:,1], c = '#729dcf', s = 50, edgecolors='black', 
                zorder = len(list_cycles) + 1) # so that we see thick dots on top of the edge endpoints
    ax.set_aspect('equal')
    if return_ax :
        return ax
    if ax_was_None :
        plt.show()


# 3D with matplotlib

def plot_cycreps_3D(pts, list_cycles, pts_to_show = None, ax = None, zoom_factor = 1., return_ax = False) :
    ''' cycreps is a list of 1D cycles, i.e.
    a list of list of couples giving the 2 indices to take from pts (endpoints)'''

    xx = pts[:,0]
    yy = pts[:,1]
    zz = pts[:,2]

    ax_was_None = False
    if ax is None :
        ax_was_None = True
        fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor) # figaspect(0.5)*1.5
        ax = fig.add_subplot(projection='3d')
    
    list_colors = ['red', 'black', 'yellow', 'blue', 'grey', 'green']
    for cy, cycle in enumerate(list_cycles) :
        if len(cycle) > 0 :
            col = list_colors[np.mod(cy, len(list_colors))]
            cycle = np.array(cycle)
            for edge in cycle :
                ax.plot(xx[edge], yy[edge], zz[edge], c = col, alpha = 0.5) 

    if pts_to_show is None :
        pts_to_show = pts.copy()
    ax.scatter(pts_to_show[:,0], pts_to_show[:,1], pts_to_show[:,2], c = '#729dcf', s = 20, edgecolors='black', alpha = .3) # so that we see thick dots on top of the edge endpoints

    if return_ax :
        return ax
    if ax_was_None :
        plt.show()
        
def plot_cycreps(pts, cycreps, pts_to_show = None, ax = None, zoom_factor = 1., return_ax = False) :
    if pts.shape[1] == 2 : # 2D
        plot_cycreps_2D(pts, cycreps, pts_to_show = pts_to_show, ax = ax, return_ax = return_ax)
    elif pts.shape[1] == 3 : # 3D
        plot_cycreps_3D(pts, cycreps, pts_to_show = pts_to_show, ax = ax, zoom_factor = zoom_factor, return_ax = return_ax)
    else :
        raise ValueError("pts should be a collection of 2D or 3D points.")

        
# 3D with pyvista
def plot_cycreps_pv(pts, cycreps = None, julia_index = True, edges = None, extra_points = None,
                   point_size = 4, cycle_line_width = 5, line_width = 1, extra_point_size = 10,
                   vol = None) :

    ''' cycreps a list of list of couples giving the 2 indices to take from pts
    if in cycreps the convention is Julia i.e. julia_index = True then decrease all indices by 1 '''
    plotter = pv.Plotter(notebook=False) # external window
    
    if vol is not None : 
        ug_vol = create_ug(vol)
        plotter.add_volume(ug_vol, cmap= 'pink', opacity="sigmoid")

    point_cloud = pv.PolyData(pts)
    plotter.add_mesh(point_cloud, color='blue', point_size=point_size, render_points_as_spheres=True)
    
    if extra_points is not None :
        extra_point_cloud = pv.PolyData(extra_points)
        plotter.add_mesh(extra_point_cloud, color='red', point_size=extra_point_size, render_points_as_spheres=True)
    
    if edges is not None and len(edges) != 0 :
        lines = np.hstack([[2, *ed] for ed in edges])
        segments = pv.PolyData(pts, lines) # BUG DU SIECLE : impossible d'ecrire lines = lines
        plotter.add_mesh(segments, color='black', style = 'wireframe', line_width = line_width)    

    if cycreps is not None and len(cycreps) != 0 :
        #list_colors = ['black', 'yellow', 'blue', 'grey', 'red', 'green']
        list_colors = np.array([
                [11/256, 11/256, 11/256, 1], #black
                [255/256, 247/256, 0/256, 1], #yellow
                [12/256, 238/256, 246/256, 1], #cyan
                [189/256, 189/256, 189/256, 1], #grey
                [1, 0, 0, 1], # red
                [10/256, 10/256, 240/256, 1], # blue
                [10/256, 240/256, 10/256, 1] # green
                ])
        for ind,cyr in enumerate( cycreps ):
            lines = np.hstack([[2, cyr[i][0] - julia_index, cyr[i][1] - julia_index] for i in range(len(cyr))])
            #lines = np.hstack([[ 2, pts[cyr[i][0] - julia_index], pts[cyr[i][1]] - julia_index] for i in range(len(cyr))])
            segments = pv.PolyData(pts, lines) # BUG DU SIECLE : impossible d'ecrire lines = lines
            plotter.add_mesh(segments, color = list_colors[ind % len(list_colors)], style = 'wireframe', line_width = cycle_line_width)    
    plotter.show()
    
### PLOT IMAGES

def slices(u, figsize = (12,4), rescale = True, cmap = 'gray', save = False, title = '') :
    '''Visualize three 2D slices of a 3D volume at z = Z//3, Z//2, or 2*Z//3
    rescale = True: grays between u.min and u.max
    rescale = False: grays between 0 and 1 (black/white beyond 0/1)'''
    vmin = None ; vmax = None
    Z = u.shape[0]
    if not rescale : vmin = 0. ; vmax = 1. 
    fig, ax = plt.subplots(1, 3, figsize = figsize, sharex=True, sharey=True)
    ax[0].imshow(np.asarray(u[Z//3], dtype=np.float64), vmin = vmin, vmax = vmax, cmap = cmap)
    ax[1].imshow(np.asarray(u[Z//2], dtype=np.float64), vmin = vmin, vmax = vmax, cmap = cmap)
    ax[2].imshow(np.asarray(u[2 * Z//3], dtype=np.float64), vmin = vmin, vmax = vmax, cmap = cmap)
    fig.tight_layout()
    if title != '' : fig.suptitle(title) ; fig.savefig(title + '.png')
    plt.show()
    
def single(img, figsize = (6,6), rescale = True, cmap = 'gray') :
    'Visualize a single 2D image'
    vmin = None ; vmax = None
    if not rescale : vmin = 0.; vmax = 1.
    plt.figure(figsize = figsize)
    plt.imshow(img,vmin = vmin, vmax = vmax, cmap = cmap)
    plt.show()
    
    
    
### PLOT PREVALENCE RESULTS

# plot prevalence-augmented barcodes
import matplotlib.patches as patches
def plot_bars_PH1(PH1, scores = None, diagonal = False, delta_y = .1, delta_y_prev = .2, figpath = None) :
    # PH1 : K x 2 array of birth-death values
    # scores : K array of prevalence values (if known)

    if type(PH1) == list :
        PH1 = np.array(PH1)
    ph1 = len(PH1)
    
    if scores is None :
        scores = np.ones(ph1)
    
    if diagonal == False :
        where_distinct = PH1[:,0] < PH1[:,1]
        PH1 = PH1[where_distinct == True]
        #if prevalence is not None :
        scores = scores[where_distinct == True]

    maxi_colorbar = scores.max()
    #col = '#1A0DAB'
    
    xmax = PH1[:,1].max()
    ymax = delta_y * ph1 + delta_y_prev * scores.sum()
    
    fig = plt.figure(figsize = (8,7))
    ax = plt.gca()
    y = 0
    for i,bar in enumerate(PH1) :
        birth, death = bar
        score = scores[i]
        level = score / maxi_colorbar #min(score, maxi_colorbar) / maxi_colorbar # rescale color

        # does not work with plot because linewidth not with data units
        ###plt.plot([birth, death],[y, y], c = plt.cm.Reds(level), linewidth = score * delta_y_prev)
        rect = patches.Rectangle((birth, y), death - birth,  score * delta_y_prev,
                                 linewidth=1, edgecolor=None, facecolor=plt.cm.Reds(level))
        ax.add_patch(rect)
        y += delta_y_prev * score + delta_y

    #plt.axis('equal')
    plt.yticks([])
    percent = 5
    plt.xlim([-xmax * percent / 100,xmax * (1 + percent / 100)])
    plt.ylim([-ymax * percent / 100,ymax*(1 + percent / 100)])
    plt.ylabel('persistence bars')
    plt.xlabel('filtration value')
    
    cmap = mpl.cm.Reds
    norm = mpl.colors.Normalize(vmin=0, vmax=maxi_colorbar)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, fraction=0.025, pad=0.04)
    #legend_elements = [ Line2D([0], [0], color=col, lw=1, label='PH1') ]
    #plt.legend(handles=legend_elements)
    
    fig.tight_layout() # put it AFTER colorbar 
    
    if figpath is not None :
        plt.savefig(figpath, dpi = 300)    
    plt.show()

    
    
def plot_cycreps_prevalence_2D(pts, list_cycles, scores, maxi_colorbar = 1, plot_points = True,
                               ax = None, zoom_factor = 1., return_ax = False) :
    ''' cycreps is a list of 1D cycles, i.e.
    a list of list of couples giving the 2 indices to take from pts (endpoints)
    prevalence is a list of numbers giving the prevalence scores of the cycles'''
    
    if scores.max() > 1 :
        raise Exception('Some prevalence scores are greater than 1. Multiple matchings?') # MAY HAPPEN
    if maxi_colorbar is None :
        maxi_colorbar = scores.max() 

    xx = pts[:,0]
    yy = pts[:,1]

    ax_was_None = False
    if ax is None :
        ax_was_None = True
        fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor) # figaspect(0.5)*1.5
        ax = plt.gca()
    for cy, cycle in enumerate(list_cycles) :
        if len(cycle) > 0 :
            cycle = np.array(cycle)
            level = min(scores[cy], maxi_colorbar) / maxi_colorbar # rescale color
            ax.plot(xx[cycle.T], yy[cycle.T],  c = plt.cm.Reds( level ) )

    if plot_points :
        ax.scatter(xx, yy, c = '#729dcf', s = 20, edgecolors='black', alpha = .3) # so that we see thick dots on top of the edge endpoints
    ax.set_aspect('equal')
    
    if ax_was_None :
        #colorbar
        if scores.max() > 1 :
            raise Exception('scores greater than 1')
            
        cmap = mpl.cm.Reds
        norm = mpl.colors.Normalize(vmin=0, vmax=maxi_colorbar)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, fraction=0.02, pad=0.04)

        plt.title('Cycles of X stained by prevalence score')
        
        if return_ax : return ax
        plt.show()

def plot_cycreps_prevalence_3D(pts, list_cycles, scores, ax = None, zoom_factor = 1.) :
    ''' cycreps is a list of 1D cycles, i.e.
    a list of list of couples giving the 2 indices to take from pts (endpoints)
    prevalence is a list of numbers giving the prevalence scores of the cycles'''

    xx = pts[:,0]
    yy = pts[:,1]
    zz = pts[:,2]

    ax_was_None = False
    if ax is None :
        ax_was_None = True
        fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor) # figaspect(0.5)*1.5
        ax = fig.add_subplot(projection='3d')
    
    for cy, cycle in enumerate(list_cycles) :
        if len(cycle) > 0 :
            cycle = np.array(cycle)
            for edge in cycle :
                ax.plot(xx[edge], yy[edge], zz[edge], c = plt.cm.Reds( scores[cy] )) 

    ax.scatter(xx, yy, zz, c = '#729dcf', s = 20, edgecolors='black', alpha = .3) # so that we see thick dots on top of the edge endpoints
    ##ax.set_aspect('equal') # not possible in 3D
    if ax_was_None :
        #colorbar
        #maxi = scores.max()
        if scores.max() > 1 :
            raise Exception('scores greater than 1')
        maxi = 1 # because expected between 0 and 1
        cmap = mpl.cm.Reds
        norm = mpl.colors.Normalize(vmin=0, vmax=maxi)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, fraction=0.02, pad=0.04)

        plt.title('Cycles of X stained by prevalence score')
        plt.show()
        
def plot_cycreps_prevalence(pts, cycreps, scores, ax = None, zoom_factor = 2) :
    #if pts.shape[1] == 2 : # 2D
    #    plot_cycreps_2D(pts, cycreps, ax = ax)
    if pts.shape[1] == 3 : # 3D
        plot_cycreps_prevalence_3D(pts, cycreps, scores, ax = ax, zoom_factor = zoom_factor)
    if pts.shape[1] == 2 : # 2D
        plot_cycreps_prevalence_2D(pts, cycreps, scores, ax = ax, zoom_factor = zoom_factor)
    else :
        raise ValueError("pts should be collection of 2D or 3D points.")

### PLOT CROSS-PREVALENCE RESULTS


def plot_cross_prevalence_2D(list_X, list_cycreps, list_scores, maxi_colorbar = 1, zoom_factor = 1.,
                             savefig = False, filename = '', return_ax = False) :
    ''' list of cycreps (each cycreps is a list of 1D cycles, i.e.
    a list of list of couples giving the 2 indices to take from pts (endpoints))
    list_scores is a list of list of numbers giving the prevalence scores of the cycles'''
    
    if max( [list_scores[i].max() for i in range(len(list_scores)) ] ) > 1 :
        raise Exception('Some prevalence scores are greater than 1. Multiple matchings?') # MAY HAPPEN
    if maxi_colorbar is None :
        maxi_colorbar = max( [max(list_scores[i]) for i in range(len(list_X))] )

    fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor)
    ax = plt.gca()
    
    for i in range(len(list_X)) :
        pts = list_X[i]
        list_cycles = list_cycreps[i]
        scores = list_scores[i]
        
        xx = pts[:,0]
        yy = pts[:,1]
        for cy, cycle in enumerate(list_cycles) :
            if len(cycle) > 0 :
                cycle = np.array(cycle)
                level = min(scores[cy], maxi_colorbar) / maxi_colorbar # rescale color
                ax.plot(xx[cycle.T], yy[cycle.T], c = plt.cm.Reds( level ))

    ax.set_aspect('equal')

    #colorbar
    cmap = mpl.cm.Reds
    norm = mpl.colors.Normalize(vmin=0, vmax=maxi_colorbar)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, fraction=0.02, pad=0.04)
    #plt.title('All cycles of X_1, ..., X_N, stained by prevalence score')

    if savefig :
        plt.savefig(filename, dpi = 200)
        
    if return_ax :
        return ax
    else :
        plt.show()

def plot_cross_prevalence_3D(list_X, list_cycreps, list_scores, zoom_factor = 1., savefig = False, filename = '') :
    ''' list of cycreps (each cycreps is a list of 1D cycles, i.e.
    a list of list of couples giving the 2 indices to take from pts (endpoints))
    list_scores is a list of list of numbers giving the prevalence scores of the cycles'''

    fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor) # figaspect(0.5)*1.5
    ax = fig.add_subplot(projection='3d')
    
    for i in range(len(list_X)) :
        pts = list_X[i]
        list_cycles = list_cycreps[i]
        scores = list_scores[i]
        
        xx = pts[:,0]
        yy = pts[:,1]
        zz = pts[:,2]
        for cy, cycle in enumerate(list_cycles) :
            if len(cycle) > 0 :
                cycle = np.array(cycle)
                for edge in cycle :
                    ax.plot(xx[edge], yy[edge], zz[edge], c = plt.cm.Reds( scores[cy] )) 

        #ax.scatter(xx, yy, zz, c = '#729dcf', s = 20, edgecolors='black', alpha = .3) # so that we see thick dots on top of the edge endpoints
        ##ax.set_aspect('equal') # not possible in 3D
        
    #colorbar
    #maxi = max( [list_scores[i].max() for i in range(len(list_scores)) ] )
    if max( [list_scores[i].max() for i in range(len(list_scores)) ] ) > 1 :
        raise Exception('Some prevalence scores are greater than 1. Multiple matchings?') # MAY HAPPEN
    maxi = 1 # prevalence scores all between 0 and 1
    cmap = mpl.cm.Reds
    norm = mpl.colors.Normalize(vmin=0, vmax=maxi)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, fraction=0.02, pad=0.04)
    #plt.title('All cycles of X_1, ..., X_N, stained by prevalence score')

    if savefig :
        plt.savefig(filename, dpi = 200)
    plt.show()
        
def plot_cross_prevalence(list_X, list_cycreps, list_scores, **kwargs) :
    #if pts.shape[1] == 2 : # 2D
    #    plot_cycreps_2D(pts, cycreps, ax = ax)
    if list_X[0].shape[1] == 3 : # 3D
        plot_cross_prevalence_3D(list_X, list_cycreps, list_scores, **kwargs)
    if list_X[0].shape[1] == 2 : # 2D
        plot_cross_prevalence_2D(list_X, list_cycreps, list_scores, **kwargs)
    else :
        raise ValueError("pts should be collection of 2D or 3D points.")


# PLOT MATCHING CYCLE REPRESENTATIVES (attached to functions in matching.py/mult_matches.py)

def argsort(seq, option = 'desc'):
    # what permutation to apply to indices in order to sort seq in ascending / descending order
    if option == 'asc' :
        return sorted(range(len(seq)), reverse = False, key= seq.__getitem__)
    elif option == 'desc' :
        return sorted(range(len(seq)), reverse = True, key= seq.__getitem__)

def show_matches(X, Y, matched_X_Y, affinity_X_Y, tight_reps_X, tight_reps_Y, dim = 1,
                 zoom_factor = 3, show_together = False) :
    ''' This function displays the matches between two point-clouds X and Y after performing the matching and obtaining
    the lists: matched_X_Y and affinity_X_Y. Also needed the corresponding lists of tight_reps.'''

    Z = np.vstack((X,Y))
    arg = argsort(affinity_X_Y)
    affinity_X_Y = np.array(affinity_X_Y)[arg]
    matched_X_Y = np.array(matched_X_Y)[arg]

    if len(matched_X_Y) == 1 or not show_together :
        for match, aff in zip(matched_X_Y, affinity_X_Y) :
            print('new match')
            a,b = match

            if X.shape[1] == 2 :
                fig, axes = plt.subplots(1,2, figsize = (8,5), sharex = True, sharey = True)
                axes[0].scatter(Y[:,0], Y[:,1], alpha = .2)
                plot_cycreps(Z, [tight_reps_X[dim][a]], pts_to_show = X, ax = axes[0])
                axes[1].scatter(X[:,0], X[:,1], alpha = .2)
                plot_cycreps(Z, [tight_reps_Y[dim][b]], pts_to_show = Y, ax = axes[1])
                for ax in axes:
                    ax.set_aspect('equal')
                axes[0].set_xlabel('X')
                axes[1].set_xlabel('Y')
                #plt.text(2, 5, 'a match with affinity {}'.format(aff))
                fig.suptitle('a match with affinity {}'.format(aff), y=0.78)
                plt.show()

            if X.shape[1] == 3 :
                fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor) # figaspect(0.5)*1.5
                fig.suptitle('a match with affinity {}'.format(aff), y=0.7)
                ax = fig.add_subplot(1,2,1, projection='3d')
                ax.scatter(Y[:,0],Y[:,1],Y[:,2], alpha = .1)
                plot_cycreps(Z, [tight_reps_X[dim][a]], pts_to_show = X, ax = ax)
                ax.set_title('X')

                ax = fig.add_subplot(1,2,2, projection='3d')
                ax.scatter(X[:,0],X[:,1],X[:,2], alpha = .1) #, c = '#729dcf', s = 50, edgecolors='black')
                plot_cycreps(Z, [tight_reps_Y[dim][b]], pts_to_show = Y, ax = ax)
                ax.set_title('Y')
                plt.show()

    else : # show_together :

        if X.shape[1] == 2 :
            fig, axes = plt.subplots(len(matched_X_Y), 2, figsize = (10, 6 * len(matched_X_Y)),
                                                                     sharex = True, sharey = True)
            # will be buggy if len(matched_X_Y) == 1
            i = 0
            for match, aff in zip(matched_X_Y, affinity_X_Y) :
                a,b = match

                axes[i,0].scatter(Y[:,0], Y[:,1], alpha = .2)
                plot_cycreps(Z, [tight_reps_X[dim][a]], pts_to_show = X, ax = axes[i,0])
                axes[i,1].scatter(X[:,0], X[:,1], alpha = .2)
                plot_cycreps(Z, [tight_reps_Y[dim][b]],  pts_to_show = Y, ax = axes[i,1])

                axes[i,0].set_xlabel('X')
                axes[i,0].set_title('aff =')
                axes[i,1].set_title(aff)
                axes[i,1].set_xlabel('Y')
                #plt.text(2, 5, 'a match with affinity {}'.format(aff))
                #fig.suptitle('a match with affinity {}'.format(aff), y=0.78)
            for ax in axes.ravel():
                ax.set_aspect('equal')
            plt.show()


        if X.shape[1] == 3 :
            fig, axes = plt.subplots(len(matched_X_Y), 2, subplot_kw={'projection': '3d'},
                                    figsize = (10, 4.5 * len(matched_X_Y)))
            #fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor) # figaspect(0.5)*1.5
            #fig.suptitle('all matches', y=0.7)

            i = 0
            for match, aff in zip(matched_X_Y, affinity_X_Y) :
                a,b = match
                #print('a,b=',a,b)
                ax = axes[i,0]
                #ax = fig.add_subplot(i,2,1, projection='3d')
                ax.scatter(Y[:,0],Y[:,1],Y[:,2], alpha = .1)
                plot_cycreps(Z, [tight_reps_X[dim][a]], pts_to_show = X, ax = ax)
                ax.set_title('X   aff = {}'.format(aff))

                ax = axes[i,1]
                #ax = fig.add_subplot(i,2,2, projection='3d')
                ax.scatter(X[:,0],X[:,1],X[:,2], alpha = .1) #, c = '#729dcf', s = 50, edgecolors='black')
                plot_cycreps(Z, [tight_reps_Y[dim][b]], pts_to_show = Y, ax = ax)
                ax.set_title('Y  a = {} b = {}'.format(a,b))

                i += 1

            plt.tight_layout()
            plt.show()
