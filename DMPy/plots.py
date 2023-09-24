#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example script on how to analysis and plot the pipeline simulations.

Author: Rik van Rosmalen"""
from __future__ import division
from __future__ import print_function

import functools
import glob

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib
# matplotlib.use('Agg')  # 'Agg' is just one of the possible backends.
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

# Use seaborn to set styles
matplotlib.rc('text', usetex=True)
sns.set_style("ticks")  # ticks -- keep axis ticks, white background, no grid
# paper -- Use predefined style for paper figures, smaller labels, thinner lines.
# However, we do want a little bit bigger text.
sns.set_context("paper", font_scale=1.2)
# Manually set the title and axislabel a tad bigger still.
plt.rcParams['axes.titlesize'] *= 1.3
plt.rcParams['axes.labelsize'] *= 1.3

# Note that on headless servers, a different back-end might need to be selected.
# You can do this with replacing the matplotlib imports with the following:
# import matplotlib
# matplotlib.use('Agg')  # 'Agg' is just one of the possible backends.
# import matplotlib.pyplot as plt  # Make sure this after changing the backend!
# from matplotlib import animation


def process_scores(path):
    """Takes the results array and processes it into a more convenient format."""
    results = np.load(path)
    # Sort the rows on the values in the columns
    idx = np.lexsort([results[:, i] for i in range(results.shape[1]-1, -1, -1)])
    results = results[idx]
    # We go from a simple 2d array of number of simulations x 4. to an alternative format
    # where we save the info per run:
    # [(fraction, np.array(scores)), ...]
    current_fraction, current_fs, current_core, current_ds, _ = results[0]
    scores = []
    for fraction, fs, core, ds, score in results:
        if fraction != current_fraction or current_core != core or current_fs != fs:
            yield current_fraction, np.array(scores)
            scores = [score]
        else:
            scores.append(score)
        current_fraction, current_fs, current_core = fraction, fs, core
    yield current_fraction, np.array(scores)


def score_convergence(results, converged=0.05, max_length=1000):
    for fraction, scores in results:
        convergence = scipy.stats.sem(scores, nan_policy='omit') / np.nanmean(scores)
        length = len(scores)
        yield (fraction, convergence, length)


def filter_converged(results, converged=0.05):
    for fraction, scores in results:
        if scipy.stats.sem(scores, nan_policy='omit') / np.nanmean(scores) < converged:
            yield fraction, scores


def scores_to_stats(results, functions=(len, np.mean, np.std, np.median, scipy.stats.sem)):
    # Keep in mind nan's can propagate!
    for fraction, scores in results:
        stats = [fraction]
        for f in functions:
            stat = f(scores)
            try:
                stats.extend(list(stat))
            except TypeError:
                stats.append(stat)
        yield stats


def bin_by_fraction(results):
    results = np.array(list(results))
    fractions = np.unique(results[:, 0])
    for i in fractions:
        yield i, results[np.where(results[:, 0] == i)][:, 1:]


def animate(nframe, ax, runs, trajectories, t_p, reference_p, metabolite_selection, colors):
    fraction = np.unique(runs[:, 0])[nframe]
    index = np.where(runs[:, 0] == fraction)[0]

    ax.cla()
    name = "{} - {}".format(file[:-4], fraction)
    ax.set_title(name)
    for i, color in zip(metabolite_selection, colors):
        ax.plot(t_p, trajectories[index][:, :, i].T, color, alpha=0.1)
    # Do the reference last so it goes on top.
    ax.plot(t_p, reference_p[:, metabolite_selection], color='black')

    ax.set_ylim(0, np.ceil(reference_p[:, metabolite_selection].max()) * 5)
    ax.set_xlim(0, t_p.max())


def savefig(fig, path, size_mm, settings):
    inch_to_mm = 25.4
    w, h = [(i / inch_to_mm) for i in size_mm]
    fig.savefig(path + '.png', **settings)
    fig.savefig(path + '.pdf', **settings)


if __name__ == '__main__':
    outpath = 'test_data/example/testing/plots/feedback_christian/'
    fig_settings = {'dpi': 300}
    save = True

    plots = {'1', '2', '3', '4a', '4b', '5a', '5b', '6', '7a', '7b', '7c', '8', '9', '10', '11',
             '12', '13', '14', '15', '16', '17'}
    skip = {'1', '2', '3', '4a', '4b'}
    to_plot = plots - skip
    to_plot = {'8'}
    # Figure 1: An overview of the computational framework
    # Done in LateX

    # Figure 2: Automated search for parameter values
    # Done in Latex

    # Table 1 : Overview of parameters found with/without identifier mapping.
    if 't1' in to_plot:
        paths = [
                 ('Costa', 'test_data/example/testing/costa/parameters_pipeline.tsv',
                  'test_data/example/testing/pipeline_bs_only/parameters_pipeline_bs_costa.tsv'),
                 ('iJO1366', 'test_data/example/testing/iJO1366/parameters_pipeline.tsv',
                  'test_data/example/testing/pipeline_bs_only/parameters_pipeline_bs_iJO1366.tsv'),
                 ('iTO977', 'test_data/example/testing/iTO977/parameters_pipeline.tsv',
                  'test_data/example/testing/pipeline_bs_only/parameters_pipeline_bs_iTO977.tsv'),
                 ('Coli Core', 'test_data/example/testing/colicore/core/parameters_pipeline.tsv',
                  'test_data/example/testing/pipeline_bs_only/parameters_pipeline_bs_core.tsv'),
                ]

        parameters = [
                     ('km', 'Michaelis constant'),
                     ('kcat', 'substrate catalytic rate constant'),
                     ('keq', 'equilibrium constant'),
                     ]

        for model, path_all, path_bs in paths:
            for path in (path_all, path_bs):
                print('-'*40)
                df = pd.read_csv(path, sep='\t')
                for sname, name in parameters:
                    # 1:3 is reaction and metabolite id
                    idx = [1, 2, 9]
                    array = df[df.QuantityType == name].values[:, idx].astype(str)
                    if array.size > 0:
                        # Split the last part of the array into source and id
                        new = np.array([i.split(None, 1) for i in array[:, 2]])
                        # Merge new source array back
                        array[:, 2] = new[:, 0]
                        total_count = len(np.unique(array[:, :2], axis=0))
                        total_count_duplicates = len(array)
                    else:
                        total_count = total_count_duplicates = 0

                    print(model, sname, total_count, total_count_duplicates)
                    if array.size > 0:
                        for db in np.unique(array[:, 2], axis=0):
                            idx = np.where(array[:, 2] == db)[0]
                            count = len(np.unique(array[idx], axis=0))
                            count_duplicates = len(array[idx])
                            print('\t', db, count, count_duplicates)

    # Figure 3: Obtained prior and posterior distributions for parameters in the L. lactis system
    # TODO: Same code as the E. coli version.

    # Figure 4: Trajectories with / without compartments / regulation
    # Version based on shaded areas of percentile of trajectory values.
    # Pro - Easy glance value of spread
    # Con - No insight into dynamics, only min/max.
    if '4a' in to_plot:
        base_path = '/Users/Rik/Code/pipeline/pipeline/test_data/example/random_model/'

        trajectory_paths = ['base.npz',
                            'comp.npz',
                            'reg.npz',
                            'comp_reg.npz']

        metabolite_selection = 0
        colors = ['r', 'g', 'b']
        q = [10, 25, 49, 51, 75, 90]
        q = np.arange(10, 90+2, 2)

        # First grab all data
        data = []
        for file in trajectory_paths:
            archive = np.load(base_path + file)
            t = archive['time']
            reference = archive['reference']
            trajectories = archive['trajectory']
            runs = archive['run']

            # Trajectories and time can have smaller steps, so get sampling interval and resample.
            interval = int(t.size / trajectories[0, :, 0].size)
            t_p = t[::interval]
            reference_p = reference[::interval]

            data.append((file, runs, t_p, reference_p))

        # Now make all plots
        fractions = np.unique(runs[:, 0])
        for fraction in fractions:
            fig, axes = plt.subplots(2, 2)
            for (file, runs, t_p, reference_p), ax in zip(data, axes.flat):
                fraction_indeces = np.where(runs[:, 0] == fraction)
                trajectories_fraction = trajectories[fraction_indeces]

                # Grab percentiles
                percentiles = np.percentile(trajectories_fraction[:, :, metabolite_selection],
                                            q, axis=0)

                # Plot
                for i in range(len(q)//2):
                    ax.fill_between(t_p, percentiles[i], percentiles[-i],
                                    color='r', alpha=0.05, linewidth=0.0)
                ax.plot(t_p, reference_p[:, metabolite_selection], c='black', alpha=0.8,
                        linewidth=0.5, linestyle='--')
                ax.set_title("{} - {:.1f}".format(file.split('.')[0], fraction))
            sns.despine(fig)

    if '4b' in to_plot:
        # Trajectories -- Alternative version that can be animated based on semi transparent lines.
        # Pro - Shows more insight in dynamics
        # Con - Can be messy.
        base_path = '/Users/Rik/Code/pipeline/pipeline/test_data/example/random_model/'

        trajectories = ['base.npz',
                        'comp.npz',
                        'reg.npz',
                        'comp_reg.npz']

        metabolite_selection = [0, 1]
        colors = ['r', 'g', 'b']

        gif = False

        for file in trajectories:
            archive = np.load(base_path + file)
            t = archive['time']
            reference = archive['reference']
            trajectories = archive['trajectory']
            runs = archive['run']

            # Trajectories and time can have smaller steps, so get sampling interval and resample.
            interval = int(t.size / trajectories[0, :, 0].size)
            t_p = t[::interval]
            reference_p = reference[::interval]
            fig, ax = plt.subplots()
            animate_f = functools.partial(animate, ax=ax, runs=runs, trajectories=trajectories,
                                          t_p=t_p, reference_p=reference_p,
                                          metabolite_selection=metabolite_selection,
                                          colors=colors)
            if gif:
                anim = animation.FuncAnimation(fig, animate_f, len(np.unique(runs[:, 0])))
                sns.despine(fig)
                if save:
                    anim.save(outpath + '{}.gif'.format(file[:-4]), writer='imagemagick', fps=2)
            elif not gif:
                for i, fraction in enumerate(np.unique(runs[:, 0])):
                    name = "{} - {}".format(file[:-4], fraction)
                    animate_f(i)
                    sns.despine(fig)
                    if save:
                        savefig(outpath + 'trajectories/' + name, **fig_settings)

    if {'5a', '5b', '6', '7a', '7b', '7c'} & to_plot:
        # path = 'test_data/example/testing/costa/results_10000.npy'
        # path = 'test_data/example/testing/costa_all_reactions/results.npy'
        # path = 'test_data/example/testing/random/results.npy'
        # path = 'test_data/example/testing/costa_di/results.npy'
        # path = 'test_data/example/testing/random_fixed_rng/results.npy'
        path = 'test_data/example/testing/costa_testing/fixed_ref_rng/results.npy'
        filtered = list(filter_converged(process_scores(path)))

        percentiles = [0, 5, 25, 50, 75, 95, 100]
        percentile_f = functools.partial(np.nanpercentile, q=percentiles)
        binned_percentiles = list(bin_by_fraction(scores_to_stats(filtered,
                                                                  functions=[percentile_f])))
        fractions = [fraction for fraction, _ in binned_percentiles]
        mean = np.vstack([(np.mean(x, axis=0)) for fraction, x in binned_percentiles]).T
        se = np.vstack([(scipy.stats.sem(x, axis=0)) for fraction, x in binned_percentiles]).T

    # Figure 5a: Mean value of median error per run at increasing amount of data
    if '5a' in to_plot:
        fig, ax = plt.subplots()
        for m, s in zip(mean, se):
            ax.errorbar(fractions, m, yerr=s)
        # ax.set_title('Mean (standard error)')
        sns.despine(fig)
        if save:
            savefig(fig, outpath + '5a', (85, 225), fig_settings)

    # Figure 5b: Median value of median error per run at increasing amount of data
    if '5b' in to_plot:
        median = np.vstack([(np.median(x, axis=0)) for fraction, x in binned_percentiles]).T
        fig, ax = plt.subplots()
        for m, s in zip(median, se):
            ax.errorbar(fractions, m, yerr=s)
        # ax.set_title('Median (standard error)')
        sns.despine(fig)
        if save:
            savefig(fig, outpath + '5b', (85, 225), fig_settings)

    # Figure 6: Histogram of simulation error distributions at increasing amount of data.
    if '6' in to_plot:
        log = True
        fraction_y_axis = True
        binned_means = list(bin_by_fraction(scores_to_stats(filtered, functions=[np.nanmean])))
        fig, axes = plt.subplots(len(binned_means), sharex=True)
        extremes = []
        for fraction, scores in binned_means:
            extremes.append(np.percentile(scores, [5, 95]))
        if log:
            extremes = np.log10(extremes)
            bins = np.logspace(np.floor(extremes.min(axis=0)[0]), np.ceil(extremes.max(axis=0)[1]),
                               num=50, base=10)
        else:
            extremes = np.array(extremes)
            bins = np.linspace(np.floor(extremes.min(axis=0)[0]), np.ceil(extremes.max(axis=0)[1]),
                               num=50)

        for ax, (fraction, scores) in zip(axes, binned_means):
            text_size = ax.yaxis.get_majorticklabels()[0].get_size()
            ax.text(1.02, 0.5, str(fraction),
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=-90,
                    transform=ax.transAxes,
                    size=text_size)
            n, bins, _ = ax.hist(scores, bins=bins, alpha=0.5)
            # Decrease tick size for left y-axis tick labels
            for l in ax.yaxis.get_majorticklabels():
                l.set_size(text_size / 1.2)
            # np.save('results/hist_n_{}'.format(fraction), n)
            # np.save('results/bin_n_{}'.format(fraction), bins)
            if fraction_y_axis:
                def helper(total):
                    def fraction_formatter(x, post):
                        return "{:.2f}".format(x / total)
                    return fraction_formatter
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(helper(scores.size)))
            if log:
                ax.set_xscale('log')
            if fraction == 0.6:
                ax.set_ylabel('Fraction of Successful Simulations')

        text_size = ax.yaxis.label.get_font_properties().get_size()
        fig.text(.95, 0.5, 'Fraction of Parameters Known',
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-90,
                 transform=fig.transFigure,
                 size=text_size)

        ax.set_xlabel('Mean Error Score')
        ax.set_xbound(lower=1e-4, upper=1e2)

        # Give a little more space between figures to avoid overlap
        # of the labels on the y-axis.
        fig.subplots_adjust(hspace=0.3)

        sns.despine(fig)
        if save:
            savefig(fig, outpath + 'costa6', (85, 225), fig_settings)

    # Figure 7: Convergence speed at increasing amount of data.
    if '7a' in to_plot:
        binned_lengths = list(bin_by_fraction(scores_to_stats(filtered, functions=[len])))
        mean = np.vstack([(np.mean(x, axis=0)) for fraction, x in binned_lengths])
        se = np.vstack([(scipy.stats.sem(x, axis=0)) for fraction, x in binned_lengths])
        std = np.vstack([(np.std(x, axis=0)) for fraction, x in binned_lengths])
        # np.save('results/conv_mean', mean)
        # np.save('results/conv_se', se)
        # np.save('results/conv_std', std)
        fig, ax = plt.subplots()
        ax.errorbar(fractions, mean, yerr=se)
        # ax.set_title('Convergence (standard error)')
        sns.despine(fig)
        if save:
            savefig(fig, outpath + '7a', (85, 225), fig_settings)

    # Figure 7: Fraction or unconverged / discarded runs.
    if {'7b', '7c'} & to_plot:
        convergence_scores = list(bin_by_fraction(score_convergence(process_scores(path))))
        fractions = []
        unconverged_fractions = []
        failed_fractions = []
        converged_fractions = []
        # Histogram of convergence scores of unconverged runs per fraction
        fig, axes = plt.subplots(len(convergence_scores), sharex=True, sharey=True)
        bins = np.linspace(0.0, 1.0, num=50)
        for ax, (fraction, scores) in zip(axes.flat, convergence_scores):
            # Start with nans to filter since they'll mess up all comparisons.
            nans = np.isnan(scores[:, 0])
            # Converged is converged and not nan
            converged = np.logical_and(scores[:, 0] <= 0.05,
                                       np.logical_not(nans))
            # Unconverged but not failed: to high p but 1000 runs and not nan
            unconverged = np.logical_and(np.logical_and(scores[:, 0] > 0.05,
                                                        scores[:, 1] >= 1000),
                                         np.logical_not(nans))
            # Failed: nan or unconverged and unfinished runs
            failed = np.logical_or(nans,
                                   np.logical_and(scores[:, 0] > 0.05,
                                                  scores[:, 1] < 1000))

            fractions.append(fraction)
            unconverged_fractions.append(unconverged.mean())
            failed_fractions.append(failed.mean())
            converged_fractions.append(converged.mean())

            # ax.set_title(str(fraction), x=.95, y=.95)
            ax.hist(scores[unconverged, 0], bins=bins)

        ax.set_xlim(0, 1)
        sns.despine(fig)
        if save and '7b' in to_plot:
            savefig(fig, outpath + '7b', (85, 225), fig_settings)

        # np.save('results/fail_fractions', fractions)
        # np.save('results/fail_unconverged_fractions', unconverged_fractions)
        # np.save('results/fail_failed_fractions', failed_fractions)
        # np.save('results/fail_converged_fractions', converged_fractions)

        if '7c' in to_plot:
            # Fraction of unconverged per fraction
            fig, ax = plt.subplots()
            ax.plot(fractions, unconverged_fractions, label='unconverged')
            ax.plot(fractions, failed_fractions, label='failed')
            ax.plot(fractions, converged_fractions, label='converged')
            ax.set_xlabel('Fractions of Known Parameters')
            ax.set_ylabel('Fraction of failed simulations')
            ax.legend()

            sns.despine(fig)
            if save:
                savefig(fig, outpath + '7c', (85, 225), fig_settings)

    # Figure 8: Simulation of model with comparisons (reduced E. coli core / lactis versions)
    if '8' in to_plot:
        def simple_plot(basepath, data, metabolite_selection, metabolite_names, t_end):
            subsample = 1
            fig, axes = plt.subplots(3, 1, sharex=True)

            for i, (name, subpath) in enumerate(data):
                archive = np.load(glob.glob(basepath + subpath + '/trajectories/*.npz')[0])
                time = archive['time']
                trajectory = archive['reference']
                index = archive['index']
                # Select metabolites / subsample
                metabolite_indeces = np.hstack([np.where(index == met)[0]
                                               for met in metabolite_selection])
                trajectory = trajectory[::subsample, metabolite_indeces]
                time = time[::subsample]

                for j, ax in enumerate(axes):
                    ax.plot(time, trajectory[:, j])

            for i, (ax, name) in enumerate(zip(axes, metabolite_names)):
                text_size = ax.yaxis.label.get_font_properties().get_size()
                ax.text(1.02, 0.5, name,
                        horizontalalignment='center',
                        verticalalignment='center',
                        rotation=-90,
                        transform=ax.transAxes,
                        size=text_size)
                low, high = ax.get_ybound()
                lower = -(high - low) / 25
                ax.set_ybound(lower=lower)
                ax.set_xbound(lower=-5, upper=t_end)
                if i == 0:
                    ax.legend([name for name, _ in data])
                if i == 1:
                    ax.set_ylabel('Concentration (mM)')
                if i == 2:
                    ax.set_xlabel('Time (s)')

            sns.despine(fig)
            return fig, axes

        basepath = 'test_data/example/testing/colicore'
        data = [('Full', '/core'), ('Pruned', '/pruned'), ('Compressed', '/reduced')]
        metabolite_selection = ['M_g3p_c', 'M_g6p_c', 'M_dhap_c']
        metabolite_names = [i.split('_')[1].upper() for i in metabolite_selection]

        fig, axes = simple_plot(basepath, data, metabolite_selection, metabolite_names, 400)
        if save:
            savefig(fig, outpath + 'core8', (85, 225), fig_settings)

        basepath = 'test_data/example/testing/costa_trajectories'
        data = [('Original', '/original'), ('Combined', '/pipeline_original'),
                ('Pipeline', '/pipeline')]
        metabolite_selection = ['G3P', 'FBP', 'PEP']
        metabolite_names = metabolite_selection

        fig, axes = simple_plot(basepath, data, metabolite_selection, metabolite_names, 300)
        if save:
            savefig(fig, outpath + 'costa8', (85, 225), fig_settings)

    # Figure 9: Compartment volumes / regulation in generated model.
    if '9' in to_plot:
        basepath = 'test_data/example/testing/random/with_trajectories/'
        subpaths = ['base', 'comp', 'reg', 'reg_comp']
        names = ['Base Model', 'Compartments', 'Regulation', 'Compartments \& Regulation']
        metabolite = 'M_3'  # {}'.format(metabolite)
        sampling_rate = 10
        fractions = [0.9]
        fig, axes = plt.subplots(2, 2, sharex=True)

        for subpath, name, ax in zip(subpaths, names, axes.flat):
            path = basepath + subpath + '/trajectories/*.npz'
            path = glob.glob(path)
            assert len(path) == 1
            data = np.load(path[0])

            trajectories = data['trajectory']
            run = data['run']
            reference = data['reference']
            time = data['time']

            metabolite_idx = np.where(data['index'] == metabolite)[0][0]
            ax.plot(time[::sampling_rate], trajectories[:, :, metabolite_idx].T,
                    alpha=0.2, color='r')
            ax.plot(time, reference[:, metabolite_idx], color='k')
            text_size = ax.yaxis.label.get_font_properties().get_size()
            ax.set_title(name)
            ax.title.set_size(text_size / 1.1)
            ax.set_ybound(lower=0)
            ax.set_xbound(lower=time[0], upper=time[-1])

            if subpath == 'reg':
                left, bottom, width, height = [0.3, 0.2, 0.15, 0.15]
                ax = fig.add_axes([left, bottom, width, height], anchor=ax.get_anchor())
                ax.plot(time[::sampling_rate], trajectories[:, :, metabolite_idx].T,
                        alpha=0.2, color='r')
                ax.plot(time, reference[:, metabolite_idx], color='k')
                ax.set_xbound(lower=time[0], upper=time[-1])
                ax.set_ylim(bottom=0, top=.05)

        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Concentration (mM)')
        axes[1, 0].set_ylabel('Concentration (mM)')

        for ax, letter in zip(axes.flat, 'ABCD'):
            ax.text(.05, 0.95, letter,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    size='large',
                    weight='bold')

        fig.subplots_adjust(hspace=0.25)
        sns.despine(fig)
        if save:
            savefig(fig, outpath + '9', (85, 225), fig_settings)

    # Figure 10: Distribution of parameters found for the iJO1366 model (or others)
    if '10' in to_plot:
        path = 'test_data/example/testing/iJO1366/full/parameters_pipeline.tsv'
        balanced_path = 'test_data/example/testing/iJO1366/full/parameters_balanced.tsv'
        distribution_path = 'test_data/example/testing/iJO1366/full/distributions.txt'

        df = pd.read_csv(path, sep='\t')
        balanced_df = pd.read_csv(balanced_path, sep='\t').replace(np.nan, 'None')

        variables = [
                     ('km', '$k^{M}$', 'Michaelis constant'),
                     ('kcat', '$k^{cat}$', 'substrate catalytic rate constant'),
                     ('keq', '$k^{eq}$', 'equilibrium constant'),
                     # ('vmax', 'forward maximal velocity'),
                     ]

        # Load distributions from file.
        distributions = {}
        with open(distribution_path, 'r') as distribution_file:
            for line in distribution_file:
                for variable, _, QT in variables:
                    if line.startswith(variable):
                        mu, sigma = [float(i.strip()) for i in (line.split(':')[1]
                                                                    .strip(' []\n')
                                                                    .split(',')[1:])]
                        distributions[QT] = (mu, sigma)

        bins = 25
        pseudo_bins = 100

        fig, axes = plt.subplots(1, 3, sharey=True)
        for ax, (name, name_plot, QT) in zip(axes.flat, variables):
            filtered = df.loc[df['QuantityType'] == QT]
            values = filtered.Mean.as_matrix()
            if name in ('keq', 'kcat'):
                filtered = set(tuple(i) for i in filtered.as_matrix()[:, 0:2])
                a, b = 1, 3
            else:
                filtered = set(tuple(i) for i in filtered.as_matrix()[:, 0:3])
                a, b = 1, 4
            # Filter only values we already saw.
            balanced_values = []
            for row in balanced_df[balanced_df.QuantityType == QT].itertuples():
                if row[a:b] in filtered:
                    balanced_values.append(row[5])
            balanced_values = np.array(balanced_values)

            # Determine a useful x-range and split into bins
            mi, ma = np.percentile(np.log(values), [5, 95])
            space = np.logspace(np.floor(mi)-1, np.ceil(ma)+1, bins)

            # Determine weights
            weights = np.ones_like(values) / values.size
            balanced_weights = np.ones_like(balanced_values) / balanced_values.size

            # Plot values
            ax.hist(values, weights=weights,
                    bins=space, alpha=0.6, color='b', label='Prior (pipeline)')
            ax.hist(balanced_values, weights=balanced_weights,
                    bins=space, alpha=0.6, color='g', label='Posterior (balanced)')

            # Plot prior
            more_space = np.logspace(np.floor(mi)-1, np.ceil(ma)+1, pseudo_bins)
            mu, sigma = distributions[QT]
            pdf = scipy.stats.lognorm.pdf(more_space, np.exp(sigma), loc=0, scale=np.exp(mu))
            # Scale to area of 1, and scale for difference in number of bins.
            norm_pdf = pdf / np.trapz(pdf) * (pseudo_bins / bins)
            ax.plot(more_space, norm_pdf, label='Prior (pseudo)', c='red')

            ax.set_xlabel(name_plot)
            ax.set_xscale('log')
            ax.set_ybound(lower=0.0, upper=1.0)

        axes[0].legend(loc=1, prop={'size': 9}, bbox_to_anchor=(1.125, 1.0))
        axes[0].set_ylabel('Fraction of Parameters')

        sns.despine(fig)
        if save:
            savefig(fig, outpath + '10', (85, 225), fig_settings)

    # Figure 11: Distance from real parameters Costa (pipeline/balanced original vs original)
    if '11' in to_plot:
        original_parameters_path = 'test_data/models/costa_parameters.csv'
        balanced_parameters_path = ('test_data/example/testing/costa_testing/'
                                    'fixed_ref_rng/parameters_balanced.tsv')
        pipeline_parameters_path = ('test_data/example/testing/costa_testing/'
                                    'fixed_ref_rng/parameters_pipeline.tsv')

        # Load all parameters
        original_parameters = pd.read_csv(original_parameters_path, sep='\t')
        balanced_parameters = pd.read_csv(balanced_parameters_path, sep='\t')
        # This one has 'None' string instead of nan for empties
        pipeline_parameters = pd.read_csv(pipeline_parameters_path, sep='\t')
        pipeline_parameters.replace('None', np.nan, inplace=True)

        balanced_dict = {i[0:3]: i[3:5] for i in balanced_parameters.itertuples(index=False)}

        # For every parameter in balanced / pipeline, express the distance from balanced
        # as the number of standard deviations of the balanced parameter between
        # the means of the parameter and the balanced parameter.
        original_z_scores = {}
        pipeline_z_scores = {}
        parameters = [
                      'Michaelis constant',
                      # 'substrate catalytic rate constant',
                      'equilibrium constant',
                      'forward maximal velocity'
                      ]

        for df, z_scores in zip([original_parameters, pipeline_parameters],
                                [original_z_scores, pipeline_z_scores]):
            for _, row in df.iterrows():
                if row.QuantityType not in parameters:
                    continue
                elif row.QuantityType == 'substrate catalytic rate constant':
                    idx = (row.QuantityType, row.SBMLReactionID, np.nan)
                else:
                    idx = (row.QuantityType, row.SBMLReactionID, row.SBMLSpeciesID)
                # Look up mean / sd in balanced
                balanced_mean, balanced_sd = balanced_dict[idx]
                z_scores[idx] = abs(balanced_mean - row.Mean) / balanced_sd

        # Create z-plot
        fig, ax = plt.subplots()

        # Create a log space
        space = np.logspace(-2, 9, 100)
        for data, linestyle in zip((original_z_scores, pipeline_z_scores), ('-', '--')):
            lines = {}
            for parameter, color in zip(parameters, ('r', 'b', 'g')):
                lines[parameter] = np.zeros(100)
                # Check how big of a fraction of values is smaller then the log space
                values = np.array([value for key, value in data.items() if key[0] == parameter])
                for i, value in enumerate(space):
                    lines[parameter][i] = np.sum(values < value) / values.size
                # Plot each variable separately.
                ax.plot(space, lines[parameter], linestyle=linestyle, color=color)
        ax.set_xscale('log')
        ax.legend(parameters)
        ax.set_ylabel("Cumulative Fraction of parameters")
        ax.set_xlabel("Distance from balanced values (z-score)")

        sns.despine(fig)
        if save:
            savefig(fig, outpath + '11', (85, 225), fig_settings)

    # Figure 12: Trajectories at different width of error distributions
    if '12' in to_plot:
        basepath = 'test_data/example/testing/costa_trajectories/'
        subpaths = ['width10', 'width5', 'width2', 'width1', 'width05']
        width = [10, 5, 2, 1, 0.5]
        metabolite = 'G3P'
        sampling_rate = 10
        fractions = [0.5, 0.8, 0.9, 1.0]
        fraction_limits = [50, 50, 50, 50]
        fig, axes = plt.subplots(len(subpaths), len(fractions), sharex=True)

        for subpath, ax_index in zip(subpaths, range(len(subpaths))):
            path = basepath + subpath + '/trajectories/*.npz'
            path = glob.glob(path)
            all_data = [np.load(i) for i in path]

            reference = all_data[0]['reference']
            time = all_data[0]['time']
            index = all_data[0]['index']

            trajectories = []
            runs = []
            for data in all_data:
                assert np.all(time == data['time'])
                assert np.all(index == data['index'])
                assert np.all(reference == data['reference'])
                trajectories.append(data['trajectory'])
                runs.append(data['run'])

            trajectories = np.vstack(trajectories)
            runs = np.vstack(runs)
            for i, fraction in enumerate(fractions):
                fraction_idx = runs[:, 0] == fraction
                metabolite_idx = np.where(data['index'] == metabolite)[0][0]
                ax = axes[ax_index, i]
                ax.plot(time[::sampling_rate], trajectories[fraction_idx, :, metabolite_idx].T,
                        alpha=0.2, color='r')
                ax.plot(time, reference[:, metabolite_idx], color='k')
                ax.set_ylim(top=fraction_limits[i], bottom=0)
                ax.set_xlim(left=time[0], right=time[-1])
                if i != 0:
                    ax.yaxis.set_visible(False)
                if ax_index != len(subpaths) - 1:
                    ax.xaxis.set_visible(False)

        text_size = ax.yaxis.get_majorticklabels()[0].get_size()
        for i in range(len(fractions)):
            axes[0, i].text(0.5, 1.1, str(fractions[i]),
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=axes[0, i].transAxes,
                            size=text_size)

        for i in range(len(subpaths)):
            axes[i, len(fractions) - 1].text(1.1, .5, str(width[i]),
                                             horizontalalignment='center',
                                             verticalalignment='center',
                                             rotation=-90,
                                             transform=axes[i, len(fractions) - 1].transAxes,
                                             size=text_size)

        text_size = ax.yaxis.label.get_font_properties().get_size()
        fig.text(0.5, 0.035, 'Time (s)',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=fig.transFigure,
                 size=text_size)
        axes[len(subpaths) // 2, 0].set_ylabel('Concentration (mM)')

        fig.text(.95, 0.5, 'Sampling width',
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-90,
                 transform=fig.transFigure,
                 size=text_size)
        fig.text(0.5, 0.95, 'Fraction of Parameters Known',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=fig.transFigure,
                 size=text_size)

        sns.despine(fig)
        if save:
            savefig(fig, outpath + '12', (85, 225), fig_settings)

    if '13' in to_plot:
        basepath = 'test_data/example/testing/costa_trajectories/reg'
        data = [('None', '/none'), ('Specific', '/specific'),
                # ('original', '/original'), ('simple', '/simple')]
                ('Allosteric', '/complete')]
        metabolite_selection = ['Glucose', 'F6P', 'Pint']
        bounds = [(79.99, 80), (-0.00005, 0.0005), (20, 25)]
        metabolite_names = metabolite_selection

        subsample = 1
        fig, axes = plt.subplots(3, 1, sharex=True)

        for i, (name, subpath) in enumerate(data):
            archive = np.load(glob.glob(basepath + subpath + '/trajectories/*.npz')[0])
            time = archive['time']
            trajectory = archive['reference']
            index = archive['index']
            # Select metabolites / subsample
            metabolite_indeces = np.hstack([np.where(index == met)[0]
                                           for met in metabolite_selection])
            trajectory = trajectory[::subsample, metabolite_indeces]
            time = time[::subsample]

            for j, ax in enumerate(axes):
                ax.plot(time, trajectory[:, j])

        for i, (ax, name, bound) in enumerate(zip(axes, metabolite_names, bounds)):
            # ax.set_title(name)
            if i == 0:
                ax.legend([j for j, _ in data])
            if i == 1:
                ax.set_ylabel('Concentration (mM)')
            if i == 2:
                ax.set_xlabel('Time (s)')

            text_size = ax.yaxis.label.get_font_properties().get_size()
            ax.text(1.02, 0.5, name,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=-90,
                    transform=ax.transAxes,
                    size=text_size)
            ax.set_ybound(lower=bound[0], upper=bound[1])
            ax.set_xbound(lower=-5, upper=400)
        sns.despine(fig)
        if save:
            savefig(fig, outpath + '13', (85, 225), fig_settings)

    if '14' in to_plot:
        model_path = 'test_data/example/testing/iJO1366/compressed/model_kinetics.xml'
        import scipy.integrate
        import sbmlwrap
        m = sbmlwrap.Model(model_path)
        m.run_initial_assignments()
        m.inline_functions()
        m.inline_rules()

        keys = np.array(list(m.species.keys()))

        atol = 1e-12
        rtol = 1e-8
        jac = False
        jit = False
        enforce_reversibility = False
        compensate_volumes = False  # No volumes defined anyway for this model.
        disabled_metabolite_fluxes = None
        if jac:
            f, jac = m.get_ode_function(disabled_metabolite_fluxes, enforce_reversibility,
                                        compensate_volumes, jac)
        else:
            f = m.get_ode_function(disabled_metabolite_fluxes, enforce_reversibility,
                                   compensate_volumes, jac)
            jac = None
        y0 = m.get_state(array=True)

        integrator = 'lsoda'
        nsteps = 5e3
        r = scipy.integrate.ode(f).set_integrator(integrator,
                                                  atol=atol,
                                                  rtol=rtol,
                                                  nsteps=nsteps,
                                                  )
        r.set_initial_value(y0, t=0.0)
        t_end = 10
        dt = 0.001

        y = []
        t = []
        values = y0

        while r.successful() and r.t < t_end:
            t.append(r.t)
            y.append(values)
            values = r.integrate(r.t + dt)

        y = np.vstack(y)

        fig, ax = plt.subplots(1, 1)
        ax.plot(t, y)

        ax.set_ylabel('Concentration (mM)')
        ax.set_xlabel('Time (s)')
        ax.set_xlim(left=t[0], right=t[-1])
        ax.set_ylim(bottom=0)
        sns.despine(fig)

        left, bottom, width, height = [0.6, 0.6, 0.25, 0.25]
        ax = fig.add_axes([left, bottom, width, height])
        ax.plot(t, y)

        ax.set_yscale('log')
        # ax.set_ylabel('concentration (mM)')
        # ax.set_xlabel('time (s)')
        ax.set_xlim(left=t[0], right=t[-1])
        ax.set_ylim(bottom=1e-12, top=1e0)
        sns.despine(fig)

        if save:
            savefig(fig, outpath + '14', (85, 225), fig_settings)

    if '15' in to_plot:
        # Combined version of 7a
        costa_base = 'test_data/example/testing/costa_testing/fixed_ref_rng/results/'
        random_base = 'test_data/example/testing/random_high_max/results/'

        fractions = np.load(costa_base + 'fail_fractions.npy')
        names = [r'\textit{L. Lactis}', 'Random Network']
        means = [np.load(i + 'conv_mean.npy') for i in (costa_base, random_base)]
        se = [np.load(i + 'conv_se.npy') for i in (costa_base, random_base)]
        std = [np.load(i + 'conv_std.npy') for i in (costa_base, random_base)]
        colors = ('green', 'orange')

        fig, ax = plt.subplots()
        for m, s, sd, name, c in zip(means, se, std, names, colors):
            ax.errorbar(fractions, m, yerr=sd, label=name, color=c)

        ax.set_xlim(left=0.19, right=1.01)
        ax.set_ylim(top=10000, bottom=0)

        ax.set_xlabel("Fraction of Parameters Known")
        ax.set_ylabel("Number of Simulations")

        ax.legend()

        sns.despine(fig)
        if save:
            savefig(fig, outpath + '15', (85, 225), fig_settings)

    if '16' in to_plot:
        # Combined version of 7c
        costa_base = 'test_data/example/testing/costa_testing/fixed_ref_rng/results/'
        random_base = 'test_data/example/testing/random_high_max/results/'

        names = [r'\textit{L. Lactis}', 'Random Network']

        fractions = [np.load(i + 'fail_fractions.npy')
                     for i in (costa_base, random_base)]
        failed_fractions = [np.load(i + 'fail_failed_fractions.npy')
                            for i in (costa_base, random_base)]
        unconverged_fractions = [np.load(i + 'fail_unconverged_fractions.npy')
                                 for i in (costa_base, random_base)]
        converged_fractions = [np.load(i + 'fail_converged_fractions.npy')
                               for i in (costa_base, random_base)]
        colors = ('green', 'orange')

        fig, ax = plt.subplots()
        for fraction, failed, unconverged, converged, color, n in zip(fractions, failed_fractions,
                                                                      unconverged_fractions,
                                                                      converged_fractions,
                                                                      colors, names):
            ax.plot(fraction, failed, marker='*', color=color,
                    label="{} (Failed)".format(n))
            ax.plot(fraction, unconverged, marker='o', color=color,
                    label="{} (Unconverged)".format(n))
            # ax.plot(fraction, converged, marker='o')

        ax.set_xlim(left=0.19, right=1.01)
        ax.set_ylim(top=1.0, bottom=-0.01)

        ax.set_xlabel("Fraction of Parameters Known")
        ax.set_ylabel("Fraction of Simulations")

        ax.legend()

        sns.despine(fig)
        if save:
            savefig(fig, outpath + '16', (85, 225), fig_settings)

    if '17' in to_plot:
        # Figure 6 from preprocessed data.
        random_base = 'test_data/example/testing/random_high_max/results/'
        fraction_y_axis = True

        fractions = np.load(random_base + 'fail_fractions.npy')
        bins = np.logspace(-3, 5, num=50, base=10)
        fake_values = bins[:-1] + np.diff(bins) / 2
        hist = [np.load(i) for i in sorted(glob.glob(random_base + 'hist_n_*.npy'))]
        # assert all([np.allclose(np.load(i), bins)
                   # for i in sorted(glob.glob(random_base + 'bin_n_*.npy'))])

        fig, axes = plt.subplots(len(hist), sharex=True)

        for ax, fraction, scores in zip(axes, fractions, hist):
            text_size = ax.yaxis.get_majorticklabels()[0].get_size()
            ax.text(1.02, 0.5, str(fraction),
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=-90,
                    transform=ax.transAxes,
                    size=text_size)
            ax.hist(fake_values, bins=bins, weights=scores, alpha=0.5)
            ax.set_xscale('log')

            for l in ax.yaxis.get_majorticklabels():
                l.set_size(text_size / 1.2)

            if fraction_y_axis:
                def helper(total):
                    def fraction_formatter(x, post):
                        return "{:.2f}".format(x / total)
                    return fraction_formatter
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(helper(scores.sum())))
            if fraction == 0.6:
                ax.set_ylabel('Fraction of Successful Simulations')

        text_size = ax.yaxis.label.get_font_properties().get_size()
        fig.text(.95, 0.5, 'Fraction of Parameters Known',
                 horizontalalignment='center',
                 verticalalignment='center',
                 rotation=-90,
                 transform=fig.transFigure,
                 size=text_size)

        ax.set_xlabel('Mean Error Score')
        ax.set_xbound(lower=1e-3, upper=1e6)

        # Give a little more space between figures to avoid overlap
        # of the labels on the y-axis.
        fig.subplots_adjust(hspace=0.3)

        sns.despine(fig)
        if save:
            savefig(fig, outpath + '17', (85, 225), fig_settings)
