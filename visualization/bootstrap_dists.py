import re
import os
import glob
import scipy
import pickle
import argparse
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# global variables 

def_fig_size = (24, 24)

def_pattern='*X_*_L2dists'

def_heatmap_vars = ["Wp_XY", "Wp_XhatYhat_i", "Mean Wp Approximation Difference", "Wphat0_XY", "Wphat_XY", "Wphat0_XY_i"]
# def_scatter_vars = ["Xname", "Yname", "Wp_XXhat_i", "Wp_YYhat_i", "dIM_XXhat_i", "dIM_YYhat_i", "PDXhat_diag_i", "PDYhat_diag_i"]
def_scatter_vars = ["Xname", "Yname", "Wp_XXhat_i", "Wp_YYhat_i", "dIM_XXhat_i", "dIM_YYhat_i", "PDXhat_diag_i", "PDYhat_diag_i", "PDX_diag", "PDY_diag"]

####################################################################################################################
# Synthesizer functions
def summarize_pair_dists(
        input_dir, 
        output_dir,
        name_type="toy_model", 
        pattern=def_pattern,
        heatmap_vars=def_heatmap_vars, 
        scatter_vars=def_scatter_vars,
        do_rainbow=True,
        cluster_method="average",
        do_scatter=True,
        write_mode=True
        ):
    alldata_grid = pull_data(input_dir, pattern=pattern, name_type=name_type)

    if "_vs_self" in input_dir:
        self_cluster=True
    else:
        self_cluster=False

    if not os.path.isdir(output_dir):
        print(f"Warning: making new directory {output_dir}")
        os.mkdir(output_dir)

    if do_rainbow:
        xnamelist, ynamelist, valuegrid_list = _get_heatmap_inputs(alldata_grid, heatmap_vars=heatmap_vars)
        generate_rainbow_plots(
                xnamelist, 
                ynamelist, 
                valuegrid_list, 
                heatmap_vars=heatmap_vars,
                outdir=output_dir,
                name_type=name_type,
                self_cluster=self_cluster,
                cluster_method=cluster_method,
                write_mode=write_mode
                )

    if do_scatter:
        scatter_df, hue_var, style_var = _get_scatter_df(alldata_grid, scatter_vars=scatter_vars, name_type=name_type)
        generate_scatter_plots(
                output_dir, 
                scatter_df, 
                hue_var=hue_var, 
                style_var=style_var, 
                write_mode=write_mode
                )


def generate_scatter_plots(
        output_dir,
        scatter_df,
        hue_var="Noise Proportion",
        style_var="Embedding Dimension",
        write_mode=True,
        debug=True
        ):

    all_plotvars = [colname for colname in scatter_df.columns if colname.endswith("_i")]
    var_pairs = list(itertools.combinations(all_plotvars, 2))
    var_triples = []
    # var_triples = list(itertools.combinations(all_plotvars, 3))

    df_list = [df for _, df in scatter_df.groupby(scatter_df['Name'])]
    
    if debug:
        ### debugging code ###
        print("Generating scater plots...\nSpecified inputs:")
        print("output_directory:\n", output_dir)
        print("dataframe (columns):\n", scatter_df.columns)
        print("hue_var:\n", hue_var)
        print("style_var:\n", style_var)
        print(f"input dataframe split into {len(df_list)} subframes:", [list(df["Name"])[0] for df in df_list])
        print("set of plotting variable pairs:\n", var_pairs)
        print("set of plotting variable triples:\n", var_triples)
        ### debugging code ###


    for df in df_list:
        name = list(df["Name"])[0]
        for pair in var_pairs:
            scatter_plot(
                    df, pair[0], pair[1], 
                    plt_title = _construct_title((name, pair[0], pair[1]), title_type = "scatter"), 
                    outdir = output_dir,
                    style_var = style_var, 
                    hue_var = hue_var,
                    space_name = name,
                    write_mode = write_mode
                    )
        for triple in var_triples:
            scatter_plot(
                    df, triple[0], triple[1], z_var=triple[2],
                    plt_title = _construct_title((name, triple[0], triple[1], triple[2]), title_type = "scatter"), 
                    outdir = output_dir,
                    style_var = style_var, 
                    hue_var = hue_var,
                    space_name = name,
                    write_mode = write_mode
                    )



def _get_scatter_df(alldata_grid, scatter_vars=def_scatter_vars, name_type="toy_model", debug=True):
    Xvars = [var for var in scatter_vars if "X" in var]
    Yvars = [var for var in scatter_vars if "Y" in var]
    
    Xdata = [ {var: i[0][var] for var in Xvars} for i in alldata_grid ]
    Ydata = [ {var.replace('Y','X'): j[var] for var in Yvars} for j in alldata_grid[0] ]
    combined_data = Xdata + Ydata

    for data_dict in combined_data:
        name, hue, style = _parse_longname(data_dict["Xname"], name_type=name_type)
        hue_var = _name_vartype(hue, name_type=name_type)
        style_var = _name_vartype(style, name_type=name_type)
        data_dict["Xname"] = name
        data_dict[hue_var] = hue
        data_dict[style_var] = style

    scatter_df = pd.DataFrame(combined_data)
    scatter_df.rename( columns = {"Xname": "Name"}, inplace=True)
    scatter_df = scatter_df.explode( [colname for colname in scatter_df.columns if colname.endswith("_i")] )

    if debug:
        ### debugging code ###
        # print(f"First entry of 'Xdata' (of {len(Xdata)}): \n{Xdata[0]}")
        # print(f"First entry of 'Ydata' (of {len(Ydata)}): \n{Ydata[0]}")
        # print(f"First and last entries of 'combined_data': \n{combined_data[0], combined_data[-1]}")
        print(f"Data Frame containing (unpaired!) scatterplot-type data has columns: \n{scatter_df.columns} \nand takes values: \n{scatter_df}\n")
        ### debugging code ###

    return scatter_df, hue_var, style_var


def generate_rainbow_plots(
        xnamelist, 
        ynamelist, 
        valuegrid_list, 
        heatmap_vars = def_heatmap_vars,
        outdir = None,
        name_type = "toy_model",
        self_cluster = False,
        cluster_method = "average",
        write_mode = True
        ):
    for i, value_grid in enumerate(valuegrid_list):
        value_name = heatmap_vars[i]
        if value_grid.ndim > 2:
            mean_grid = np.mean(value_grid, axis=2)
            rainbow_plot(
                    f"{value_name}_mean", 
                    mean_grid, 
                    xnamelist, 
                    ynamelist,
                    outdir=outdir,
                    name_type=name_type,
                    self_cluster=self_cluster,
                    write_mode=write_mode
                    )
            std_grid = np.std(value_grid, axis=2)
            rainbow_plot(
                    f"{value_name}_stddev", 
                    std_grid, 
                    xnamelist, 
                    ynamelist,
                    outdir=outdir,
                    name_type=name_type,
                    self_cluster=self_cluster,
                    write_mode=write_mode
                    )
        else:
            rainbow_plot(
                    value_name, 
                    value_grid, 
                    xnamelist, 
                    ynamelist,
                    outdir=outdir,
                    name_type=name_type,
                    self_cluster=self_cluster,
                    write_mode=write_mode
                    )



def _get_heatmap_inputs(alldata_grid, heatmap_vars=def_heatmap_vars, debug=True):
    xnamelist = [i[0]["Xname"] for i in alldata_grid]
    ynamelist = [j["Yname"] for j in alldata_grid[0]]

    valuegrid_list = [None]*len(heatmap_vars)

    for idx, varname in enumerate(heatmap_vars):
        try:
            valuegrid_list[idx] = np.array([[j[varname] for j in i] for i in alldata_grid])
        except ValueError:
            new_entry = [np.array([j[varname] for j in i]) for i in alldata_grid]
            valuegrid_list[idx] = new_entry
            if debug:
                ### debugging code ###
                print(f"found data inmogeneity in {varname} readin. attempted new entry has data of following shapes and values:")
                print([var.shape for var in new_entry])
                print("corresponding to pairs:")
                print([[(j["Xname"],j["Yname"]) for j in i] for i in alldata_grid])
                # print(new_entry)
                ### debugging code ###
            

    if debug:
        ### debugging code ###
        print(f"Names of 'X' spaces: \n{xnamelist}")
        print(f"Names of 'Y' spaces: \n{ynamelist}")
        print(f"Entries in list of grid values have the following shapes: \n{[i.shape for i in valuegrid_list]}")
        print(f"Generating one heatmap for each of the following set of variables: \n{heatmap_vars}")
        print("")
        ### debugging code ###
    return xnamelist, ynamelist, valuegrid_list
####################################################################################################################


####################################################################################################################
# Figure plotting functions
def scatter_plot(
        dataframe, x_var, y_var, z_var=None, space_name=None,
        write_mode=True, plt_title = None, outdir=None,
        style_var="noise_lvl", style_order=None,
        hue_var="emb_dim", hue_order=None
        ):

    if z_var is None:
        fig, ax = plt.subplots()
        g = sns.scatterplot(
                data = dataframe,
                x = x_var,
                y = y_var,
                markers = True,
                style = style_var,
                style_order = style_order,
                hue = hue_var,
                hue_order = hue_order,
                legend = "brief"
                )
        g.set(xlabel = x_var)
        g.set(ylabel = y_var)
        g.set(title = plt_title)
    elif isinstance(z_var, str):
        cmap = matplotlib.colors.ListedColormap(sns.color_palette("Spectral", 256).as_hex())    
        fig, ax = plt.subplots()
        ax = fig.add_subplot(projection = '3d')
        mdict = _manual_stylemap(dataframe[style_var].drop_duplicates().sort_values(), style_order=style_order)
        cdict = _manual_colormap(dataframe[hue_var].drop_duplicates().sort_values())
        sc = ax.scatter(
                dataframe[x_var],
                dataframe[y_var],
                dataframe[z_var],
                marker = list(map(mdict.get, list(dataframe[style_var]))),
                c = list(map(cdict.get, list(dataframe[hue_var]))),
                cmap = cmap
                )
        ax.set_xlabel( x_var )
        ax.set_ylabel( y_var )
        ax.set_zlabel( z_var )
    
    if write_mode:
        outpath = os.path.join(outdir, f"scatter_{space_name}_x-{x_var}_y-{y_var}_hue-{hue_var}_sty-{style_var}.png").replace(" ","")
        if z_var is not None:
            outpath = outpath.replace("_hue-", f"_z-{z_var}_hue-")
        _write_img(fig, outpath)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()
    

def rainbow_plot(
        value_name, 
        value_grid, 
        xnamelist, 
        ynamelist, 
        outdir=None,
        name_type="toy_model", 
        self_cluster=True,
        cluster_method="average",
        write_mode=True,
        debug=True
        ):

    print(f"Plotting grid of '{value_name}' values...")

    xname,_,_ = _parse_longname(xnamelist[0], name_type=name_type)
    yname,_,_ = _parse_longname(ynamelist[0], name_type=name_type)
    xticklabels = ["\n".join(_parse_longname(i, name_type=name_type)[1:]) for i in xnamelist]
    yticklabels = ["\n".join(_parse_longname(i, name_type=name_type)[1:]) for i in ynamelist]
    vartype = _get_vartype(xticklabels[0].split('\n'), name_type=name_type)
    rb_title = _construct_title((value_name, xname, yname, vartype), title_type="heatmap")

    from figs_compare_topostats import _plot_clustermap

    if debug:
        ### debugging code ###
        print(f"xticklabels: \n{xticklabels}")
        print(f"yticklabels: \n{yticklabels}")
        ### debugging code ###

    if self_cluster:
        xlinkage=None
        if debug:
            ### debugging code ###
            value_uniq = np.unique((value_grid + value_grid.T)/2)
            print(f"Variable 'value_grid' has shape: {value_grid.shape}")
            print(f"Variable 'value_grid' is roughly symmetric: {np.allclose(value_grid, value_grid.T)}.")
            print(f"Variable 'value_grid' is strictly symmetric: {np.all(np.equal(value_grid, value_grid.T))}.")
            print(f"Variable 'value_grid' contains {len(value_uniq)} unique elements:")
            print(np.histogram(value_uniq))
            print("")
            ### debugging code ###

        if np.allclose(value_grid, value_grid.T):
            value_grid = (value_grid + value_grid.T)/2      # force exact symmetry because clustermap symmetry tolerance is stricter than np.allclose
        else:
            raise ValueError("Variable 'value_grid' must be (at least approximately) symmetric to use as linkage for a clustermap")
    else:
        import scipy.cluster.hierarchy as hc
        xlinkage = hc.linkage(value_grid, method=cluster_method, optimal_ordering=True)

    g = _plot_clustermap(
        value_grid, 
        cluster=True,
        cluster_method=cluster_method,
        cm_title = rb_title,
        xticklabels=xticklabels, 
        yticklabels=yticklabels,
        xlinkage=xlinkage,
        ylinkage=None,
        cmap = sns.color_palette("Spectral", as_cmap=True),
        fig_size=def_fig_size,
        write_mode=False,
        debug=debug
        )

    fig = g.fig
    ax = g.ax_heatmap
    ax.set(xlabel = xname)
    ax.set(ylabel = yname)

    if write_mode:
        outpath = os.path.join(outdir, f"rainbow_cluster_{xname}_{yname}_{value_name}.png").replace(" ","")
        _write_img(fig, outpath)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()

    fig, ax = plt.subplots()
    ax = sns.heatmap(
            value_grid, 
            square = True, 
            cbar = True, 
            ax=ax, 
            cmap = sns.color_palette("Spectral", as_cmap=True),
            xticklabels=xticklabels, 
            yticklabels=yticklabels
            )
    ax.set(title = rb_title)
    ax.xaxis.tick_top()
    ax.set(xlabel = xname)
    ax.set(ylabel = yname)

    if write_mode:
        outpath = os.path.join(outdir, f"rainbow_heatmap_{xname}_{yname}_{value_name}.png").replace(" ","")
        _write_img(fig, outpath)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()

def _construct_title(nametuple, title_type=None):
    if title_type=="heatmap":
        title = "Variation of %s \n in %s and %s \n over %s pairs" % nametuple
    elif title_type=="scatter":
        if len(nametuple) == 3:
            title = "Relationships between resampling stability measurements in %s:\n%s vs. %s" % nametuple
        elif len(nametuple) == 4:
            title = "Relationships between resampling stability measurements in %s:\n%s vs. %s vs. %s" % nametuple
        else:
            raise IOError(f"Input 'nametuple'={nametuple} must have len=3 or len=4 for title of type {title_type}")
    else:
        raise IOError(f"Unrecognized title type '{title_type}'")
    return title

def _get_vartype(ticklabel, name_type="toy_model", debug=True):

    if debug:
        ### debugging code ###
        print(f"Submitting ticklabels that look like: \n{ticklabel}")
        ### debugging code ###

    if isinstance(ticklabel, list):
        vartype = [None]*len(ticklabel)
        for i,name in enumerate(ticklabel):
            vartype[i] = _name_vartype(name, name_type=name_type)
        vartype = tuple(vartype)
    else:
        vartype = _name_vartype(ticklabel, name_type=name_type)
    return vartype

def _name_vartype(name, name_type="toy_model", debug=True):
    if debug:
        ### debugging code ###
        print(f"Producing variable type from name '{name}' under '{name_type}' conventions.")
        ### debugging code ###
    if name_type=="toy_model":
        if "e-" in name or "noiseless" in name:
            vartype = "Noise Proportion"
        elif re.search(r"d[0-9]{1,4}", name):
            vartype = "Embedding Dimension"
        else:
            raise Exception(f"Unable to assign variable type to variable '{name}'")
    elif name_type=="exp_results":
        if "Psim" in name or name=="inner" or name=="geodesic":
            vartype = "Dissimilarity Function"
        elif name=="Maps" or name=="Amps" or name=="pNMs" or name=="NMs":
            vartype = "Feature Type"
    else:
        raise IOError("Unrecognized name_type {name_type}")

    if debug:
        ### debugging code ###
        print(f"variable type name: {vartype}")
        ### debugging code ###
    return vartype

def _manual_stylemap(uniq_data, style_order=None, debug=True):
    if style_order is None:
        style_order = list(matplotlib.markers.MarkerStyle("").markers.keys())
    mdict = {}
    for i,val in enumerate(uniq_data):
        mdict[val] = style_order[i]

    if debug:
        ### debugging code ###
        print("Style order:", style_order)
        print("Marker dictionary:", mdict)
        ### debugging code ###
    return mdict

def _manual_colormap(uniq_data, debug=True):
    cvec = np.linspace(0, 1, len(uniq_data))
    cdict = {}
    for i, val in enumerate(uniq_data):
        cdict[val] = cvec[i]
    
    if debug:
        ### debugging code ###
        print("Color dictionary:", cdict)
        ### debugging code ###
    return cdict
#######################################################################################################################



#######################################################################################################################
# Data wrangling functions
def pull_data(parent_dir, pattern='*X_*_L2dists', name_type="toy_model", debug=True):
    dirlist = glob.glob(os.path.join(parent_dir, pattern))
    dirlist.sort()

    fpath_grid = [ glob.glob(os.path.join(X_dir, f"{pattern}.pkl")) for X_dir in dirlist ]
    [i.sort() for i in fpath_grid]

    alldata_grid = [ [ _load(fpath, name_type=name_type) for fpath in X_sublist ] for X_sublist in fpath_grid ]

    if debug:
        ### debugging code ###
        print(f"Pulling from fpath_grid \n{fpath_grid}")
        if not isinstance(alldata_grid[0], list):
            print(f"alldata_grid loadin variable is not nested lists, but instead has following structure: \n{[type(x) for x in alldata_grid]}")
        try:
            print(f"00 entry of alldata_grid: \n{alldata_grid[0][0]}")
        except IndexError:
            print(f"0-row entry of alldata_grid: \n{alldata_grid[0]}")
        ### debugging code ###
    return alldata_grid


def _load(input_fpath, name_type="toy_model"):
    with open(input_fpath, 'rb') as fin:
        data_dict = pickle.load(fin)

    xname_long, yname_long = _parse_input_fpath(input_fpath, name_type=name_type)
    data_dict["Xname"] = xname_long
    data_dict["Yname"] = yname_long

    return data_dict

def _parse_input_fpath(fpath, name_type="toy_model"):
    if name_type=="toy_model":
        fpath = fpath.replace('_L2dists','').replace('_L2scale','')
    elif name_type=="exp_results":
        fpath = fpath.replace('_dists','')
    else:
        raise IOError(f"Unrecognized name type {name_type}.")

    Yname_long = os.path.basename(fpath).replace('Wp_hat_X_vs_Y_','').split('.')[0]
    Xname_long = os.path.basename(os.path.dirname(fpath)).replace('X_','')
    return Xname_long, Yname_long

def _parse_longname(long_name, name_type="toy_model", debug=True):
    if debug:
        ### debugging code ###
        print(f"parsing name: {long_name}")
        ### debugging code ###

    if name_type=="toy_model":
        if "noiseless" in long_name:
            noise_prop = "noiseless" 
        else:
            found = re.findall(r"\_snr10e-[0-9],?[0-9]?", long_name)[0]
            noise_prop = found.replace("_snr","").replace(",",".")
            noise_prop = noise_prop.replace("10e-", "1e-")   # hacky fix to me forgetting how scientific notation is supposed to work when i named things :P
        
        emb_dim = re.findall(r"d[0-9]{1,4}", long_name)[0]
        emb_name = long_name.split('_d')[0]
        output = [emb_name, emb_dim, noise_prop]
    elif name_type=="exp_results":
        name_parts = long_name.split('_')
        emb_name = name_parts[0]
        feature = name_parts[1]
        dist = "_".join(name_parts[2:])

            ### debugging code ###
        output = [emb_name, feature, dist]
    else:
        raise IOError(f"Unrecognized name type {name_type}.")

    if debug:
        ### debugging code ###
        print(f"parsed into: {output}")
        ### debugging code ###

    return output


def _write_list(outpath, list_out):
    with open(outpath, 'w') as fout:
        fout.write(list_out.__str__())

def _write_img(fig, outpath, fig_size=def_fig_size):
    fig.set_size_inches(fig_size, forward=False)
    fig.savefig(outpath, dpi=600)
    print(f"saved to {outpath}")
#######################################################################################################################



#######################################################################################################################
# parses input, saves output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Create and write summary figures summarizing bootstrapped distance data"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="",
        help="directory with name of type []_vs_[] containing bootstrapped distance outputs"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="figure output directory"
    )
    parser.add_argument(
        "-n",
        "--name_type",
        type=str,
        default="toy_model",
        help="Specifies the class of naming conventions used for the target data"
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default=def_pattern,
        help="substring pattern to specify subset of matching directories"
    )
    parser.add_argument(
        "-R",
        "--do_rainbow",
        default=False,
        action="store_true",
        help="Generate heatmaps of pairwise summary comparisons over varying parameters in each pair"
    )
    parser.add_argument(
        "-S",
        "--do_scatter",
        default=False,
        action="store_true",
        help="Generate scatterplots of per-space stability summary quantities"
    )
    parser.add_argument(
        "-w",
        "--write_mode",
        default=False,
        action="store_true",
        help="write plots to .png"
    )
    args = parser.parse_args()

    summarize_pair_dists(
            args.input_dir, 
            args.output_dir,
            pattern=args.pattern,
            name_type=args.name_type,
            do_rainbow=args.do_rainbow,
            do_scatter=args.do_scatter,
            write_mode=args.write_mode
            )
