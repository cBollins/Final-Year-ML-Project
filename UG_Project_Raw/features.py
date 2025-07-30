import uproot, numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

def event_view(events, event_number):
    
    w_hits = np.concatenate(events.filter_by_event(events.reco_hits_w, event_number))
    x_hits = np.concatenate(events.filter_by_event(events.reco_hits_x_w, event_number))
    
    plt.figure(figsize = (8,6))
    plt.scatter(w_hits, x_hits, c='grey', s=4, label='Hits',)
    plt.title(f'Event display for event {event_number}')
    plt.ylabel('X (W View)')
    plt.xlabel('W (Wire Position)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_idx(events, event_idx):

    w_hits = events.reco_hits_w[event_idx]
    x_hits = events.reco_hits_x_w[event_idx]

    if len(w_hits) == len(x_hits) and len(w_hits) > 2:
        plt.figure(figsize = (8,6))
        plt.scatter(w_hits, x_hits, c='grey', s=4, label='Hits')
        plt.title(f'View for particle idx {event_idx}')
        plt.ylabel('X (W View)')
        plt.xlabel('W (Wire Position)')
        plt.legend()
        plt.grid(True)
        plt.show()
    else: print('Less than 3 hits')


def rms(events, event_idx, hits_cutoff=2):
    w_hits = events.reco_hits_w[event_idx]
    x_hits = events.reco_hits_x_w[event_idx]

    if len(w_hits) == len(x_hits) and len(w_hits) > hits_cutoff:
        slope, intercept = np.polyfit(w_hits, x_hits, 1)
        
        actual = x_hits
        predicted = slope * w_hits + intercept
        
        meanSquaredError = ((predicted - actual) ** 2).mean()
        return np.sqrt(meanSquaredError)
    else: return


def rms_pdg(events, event_idx, hits_cutoff=2):
    w_hits = events.reco_hits_w[event_idx]
    x_hits = events.reco_hits_x_w[event_idx]

    pdg = events.mc_pdg[event_idx]
    track_or_shower = None
    if pdg in [-11, 11, 22]:
        track_or_shower = 'S'
    else: track_or_shower = 'T'

    if len(w_hits) == len(x_hits) and len(w_hits) > hits_cutoff:
        slope, intercept = np.polyfit(w_hits, x_hits, 1)
    
        actual = x_hits
        predicted = slope * w_hits + intercept

        meanSquaredError = ((predicted - actual) ** 2).mean()
        return (np.sqrt(meanSquaredError), track_or_shower)
    else: return


def rms_pdg_histogram(start_idx, end_idx, hits_cutoff):
    if not isinstance(start_idx, int) or not isinstance(start_idx, int) or start_idx < 0 or end_idx < 0 or start_idx > end_idx:
        return print('invalid entry of indices')

    track_rms = []
    shower_rms = []
    
    for i in range(start_idx, 1 + end_idx):
        try:
            rms_value, indicator = rms_pdg(events, i, hits_cutoff)
            if rms_value is not None:
                if indicator == 'T':
                    track_rms.append(rms_value)
                elif indicator == 'S':
                    shower_rms.append(rms_value)
        except TypeError:
            pass

    bin_edges = np.arange(0, 31, 1)
    num_bins = len(bin_edges) - 1
    track_density, _ = np.histogram(track_rms, bins=np.arange(0, 31, 1), density=True)
    shower_density, _ = np.histogram(shower_rms, bins=np.arange(0, 31, 1), density=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(track_rms, bins=bin_edges, density=True, color='c', edgecolor='k', linewidth=0.5, alpha=0.7, label='Tracks')
    plt.hist(shower_rms, bins=bin_edges, density=True, color='m', edgecolor='k', linewidth=0.5, alpha=0.7, label='Showers')
    plt.title(f'Histogram of RMS Values for events above {hits_cutoff} hits')
    plt.xlabel('RMS Value')
    plt.ylabel('Density')
    plt.xticks(np.arange(0, 26, 2))
    plt.xlim(0,26)
    plt.legend()
    plt.grid(False)
    plt.show()


def find_median(arr): # finding the median so that we can look for the critical value between the medians of track and shower
    # Sort the array
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    
    # If the array length is odd, return the middle element
    if n % 2 == 1:
        return sorted_arr[n // 2]
    # If the array length is even, return the average of the two middle elements
    else:
        mid1, mid2 = sorted_arr[n // 2 - 1], sorted_arr[n // 2]
        return (mid1 + mid2) / 2


def rms_kde(events, start_idx, end_idx, hits_cutoff):
    # Validate inputs
    if not isinstance(start_idx, int) or not isinstance(end_idx, int) or start_idx < 0 or end_idx < 0 or start_idx > end_idx:
        return print('Invalid entry of indices')

    track_rms = []
    shower_rms = []
    
    # Loop over all particles between start/end idx inclusive
    for i in range(start_idx, 1 + end_idx):
        try:
            rms_value, indicator = rms_pdg(events, i, hits_cutoff)
            if rms_value is not None:
                if indicator == 'T':
                    track_rms.append(rms_value)
                elif indicator == 'S':
                    shower_rms.append(rms_value)
        except TypeError:
            pass

    # Define bin edges for the histogram
    bin_edges = np.arange(0, 31, 1)
    
    # Plot the histograms
    plt.figure(figsize=(10, 6))
    plt.hist(track_rms, bins=bin_edges, density=True, color='c', edgecolor='k', linewidth=0.5, alpha=0.2, label='Tracks')
    plt.hist(shower_rms, bins=bin_edges, density=True, color='m', edgecolor='k', linewidth=0.5, alpha=0.2, label='Showers')
    
    # KDE Plots using seaborn for visualization
    sns.kdeplot(track_rms, color='c', linewidth=2, label='Track KDE')
    sns.kdeplot(shower_rms, color='m', linewidth=2, label='Shower KDE')

    # Calculate KDE intersections using scipy's gaussian_kde
    track_rms_med = find_median(track_rms)
    shower_rms_med = find_median(shower_rms)
    
    x_values = np.linspace(track_rms_med, shower_rms_med, 1500)  # Shared x-range for KDEs, critical rms will be between the medians
    track_kde = gaussian_kde(track_rms)
    shower_kde = gaussian_kde(shower_rms)
    track_density = track_kde(x_values)
    shower_density = shower_kde(x_values)
    
    # Find intersection points
    intersections = x_values[np.isclose(track_density, shower_density, atol=1e-3)]
    
    # Calculate the mean of intersection points if any intersections exist
    mean_intersection = 0
    if intersections.size > 0:
        mean_intersection = intersections.mean()
        
        # Plot the mean intersection point
        plt.axvline(mean_intersection, color='dimgrey', linestyle='--', linewidth=1.2)

        y_lim = plt.ylim()
        y_text = y_lim[0] + 0.6 * (y_lim[1] - y_lim[0])
        plt.text(mean_intersection, y_text, 'Critical RMS (Mean)', rotation=90, color='dimgrey', ha='right')
    else:
        print("No intersections found.")
    
    # Finalize plot with labels and legend
    plt.title(f'KDE Overlay on RMS Histogram for events above {hits_cutoff} hits. Critical RMS: {mean_intersection.round(3) if mean_intersection is not None else mean_intersection}')
    plt.xlabel('RMS Value')
    plt.ylabel('Density')
    plt.xticks(np.arange(0, 26, 2))
    plt.xlim(0, 26)
    plt.legend()
    plt.grid(False)
    plt.show()


def track_or_shower_confidence(events, start_idx, end_idx, hits_cutoff, target_idx, show_plot):
    # Validate inputs
    if not isinstance(start_idx, int) or not isinstance(end_idx, int) or not isinstance(target_idx, int) or start_idx < 0 or end_idx < 0 or not start_idx <= target_idx <= end_idx:
        return print('Invalid entry of indices')
    
    track_rms = []
    shower_rms = []
    
    # Loop over all particles between start/end idx inclusive
    for i in range(start_idx, 1 + end_idx):
        try:
            rms_value, indicator = rms_pdg(events, i, hits_cutoff)
            if rms_value is not None:
                if indicator == 'T':
                    track_rms.append(rms_value)
                elif indicator == 'S':
                    shower_rms.append(rms_value)
        except TypeError:
            pass

    # Calculate KDE intersections using scipy's gaussian_kde
    track_rms_med = find_median(track_rms)
    shower_rms_med = find_median(shower_rms)
    
    x_values = np.linspace(track_rms_med, shower_rms_med, 1500)  # Shared x-range for KDEs, critical rms will be between the medians
    track_kde = gaussian_kde(track_rms)
    shower_kde = gaussian_kde(shower_rms)
    track_density = track_kde(x_values)
    shower_density = shower_kde(x_values)
    
    # Find intersection points
    intersections = x_values[np.isclose(track_density, shower_density, atol=1e-3)]
    
    # Calculate the mean of intersection points if any intersections exist
    rms_crit = None
    if intersections.size > 0:
        rms_crit = intersections.mean()
    else:
        print("No intersections found.")

    # Need to find the same gaussians now, but for a linspace that goes across the whole space:
    x_values_conf = np.linspace(0, 1.5*max([max(shower_rms), max(track_rms)]), 1500)
    track_conf = track_kde(x_values_conf)
    shower_conf = shower_kde(x_values_conf)
    
    target_rms = rms(events, target_idx, hits_cutoff)
    if target_rms == None:
        print(f'Cutoff is too large for event {target_idx} or an error found with calculating RMS')
        return [None, None, None]
    elif target_rms == rms_crit:
        print('RMS is the critical value, unable to discern')
        return [target_rms, 'U', 0.5] # Unknown with 50% confidence, sort of pointless informtion really.
    elif target_rms > rms_crit:
        
        # Define the limits for integration
        lower_limit = rms_crit
        upper_limit = target_rms
        
        if show_plot:
            plt.figure(figsize = (6,3))
            plt.plot(x_values_conf, track_conf, c='c', label='track')
            plt.plot(x_values_conf, shower_conf, c='m', label='shower')
            plt.axvline(rms_crit, c='dimgray', linestyle='--', label='crit rms')
            plt.axvline(target_rms, c='g', label='target rms')
            plt.title(f'KDEs for cutoff: {hits_cutoff}, Target event: {target_idx}')
            plt.legend()
            plt.show()
    
        # Create a mask to filter values within the specified limits
        mask_track = (x_values_conf >= lower_limit) & (x_values_conf <= upper_limit)
        mask_shower = (x_values_conf >= lower_limit) & (x_values_conf <= upper_limit)
    
        # Perform trapezoidal integration within the specified limits
        area_track = np.trapz(track_conf[mask_track], x_values_conf[mask_track])
        area_shower = np.trapz(shower_conf[mask_shower], x_values_conf[mask_shower])
        conf = area_shower/(area_track + area_shower)
        
        print(f'^ The RMS of event {target_idx} is {target_rms}, it is a SHOWER with a {100*(conf.round(5))}% confidence')
        return [target_rms, 'S', conf]

    elif target_rms < rms_crit:
        
        # Define the limits for integration
        lower_limit = target_rms
        upper_limit = rms_crit
        
        if show_plot:
            plt.figure(figsize = (6,3))
            plt.plot(x_values_conf, track_conf, c='c', label='track')
            plt.plot(x_values_conf, shower_conf, c='m', label='shower')
            plt.axvline(rms_crit, c='dimgray', linestyle='--', label='crit rms')
            plt.axvline(target_rms, c='g', label='target rms')
            plt.title(f'KDEs for cutoff: {hits_cutoff}, Target event: {target_idx}')
            plt.legend()
            plt.show()
    
        # Create a mask to filter values within the specified limits
        mask_track = (x_values_conf >= lower_limit) & (x_values_conf <= upper_limit)
        mask_shower = (x_values_conf >= lower_limit) & (x_values_conf <= upper_limit)
    
        # Perform trapezoidal integration within the specified limits
        area_track = np.trapz(track_conf[mask_track], x_values_conf[mask_track])
        area_shower = np.trapz(shower_conf[mask_shower], x_values_conf[mask_shower])
        conf = area_track/(area_shower + area_track)
        
        print(f'^ The RMS of event {target_idx} is {target_rms}, it is a TRACK with a {100*conf:.5f}% confidence')
        return [target_rms, 'T', conf]
    else: return


def plot_idx_adc(events, event_idx):
    
    # finding x/w hits again for a single event
    w_hits = events.reco_hits_w[event_idx]
    x_hits = events.reco_hits_x_w[event_idx]
    adc_values = events.reco_adcs_w[event_idx]

    if len(w_hits) == len(x_hits) and len(w_hits) > 5:
        plt.figure(figsize = (8,6))
        scatter = plt.scatter(w_hits, x_hits, c=adc_values, cmap='viridis', s=4, alpha=0.9) # scatter plot
        plt.colorbar(scatter, label='ADC (Energy Deposit)') # Generating Colour Bar
        plt.title(f'ADC View for particle idx {event_idx}')
        plt.ylabel('X (W View)')
        plt.xlabel('W (Wire Position)')
        plt.grid(True)
        plt.show()
    else: return print('Less that 5 hits in event, not sufficient to plot.')


def adc_max_hist(events, start_idx, end_idx, hits_cutoff):

    if hits_cutoff < 2:
        hits_cutoff = 2
        print('cannot accept events with less than 2 hits')
    # define an array for the max values that we will plot with eventually
    max_array_shower = []
    max_array_track = []

    # add a counter for the fails and successes, just so the debugging data isnt as invasive
    f = 0
    n = 0
    
    for event_idx in range(start_idx, end_idx + 1):

        # find w positions so we can relate the adc values
        w_hits = events.reco_hits_w[event_idx]

        if len(w_hits) > hits_cutoff:
            
            adc_values = events.reco_adcs_w[event_idx]
        
            # filtering track or shower from pdg code
            pdg = events.mc_pdg[event_idx]
            shower = False # initialising the track or shower variable

            if pdg in [-11, 11, 22]:
                shower = True

            # finding the w value associated with the maximum adc value
            w_idx = np.argmax(adc_values)

            # we are omitting an edge case here that there is a chance there are two identical max values, this will rarely happen though

            # want to scale the w_adc_max value
            w_adc_max = w_hits[w_idx]
            # in the samw way we scale the w hits, translate to origin then squash to [0,1]:

            w_adc_max = (w_adc_max - min(w_hits))/(max(w_hits) - min(w_hits))
            n += 1

            if shower:
                max_array_shower.append(w_adc_max)
            else: max_array_track.append(w_adc_max)

        else: f += 1

    print(f'There were {f} failed indices')
    
    edges = np.linspace(0,1,21)
    plt.figure(figsize=(10,6))
    plt.hist(max_array_track, density=True, color='c', edgecolor='k', bins=edges, linewidth=0.5, alpha=0.7, label='True Track')
    plt.hist(max_array_shower, density=True, color='m', edgecolor='k', bins=edges, linewidth=0.5, alpha=0.7, label='True Shower')
    plt.title(f'ADC Peak histogram for {n} events, cutoff = {hits_cutoff}')
    plt.xlabel('W position in event')
    plt.ylabel('ADC Max density')
    plt.legend()
    plt.grid(False)
    plt.show()


def adc_mean_hist(events, start_idx, end_idx, hits_cutoff):

    if hits_cutoff < 2:
        hits_cutoff = 2
        print('cannot accept events with less than 2 hits')
    # define an array for the max values that we will plot with eventually
    avg_array_shower = []
    avg_array_track = []

    # add a counter for the fails and successes, just so the debugging data isnt as invasive
    f = 0
    n = 0
    
    for event_idx in range(start_idx, end_idx + 1):

        # find w positions so we can relate the adc values
        w_hits = events.reco_hits_w[event_idx]

        if len(w_hits) > hits_cutoff:
            
            adc_values = events.reco_adcs_w[event_idx]
            w_adc_avg = np.mean(adc_values)
        
            # filtering track or shower from pdg code
            pdg = events.mc_pdg[event_idx]
            shower = False # initialising the track or shower variable

            if pdg in [-11, 11, 22]:
                shower = True

            n += 1

            if shower:
                avg_array_shower.append(w_adc_avg)
            else: avg_array_track.append(w_adc_avg)

        else: f += 1

    print(f'There were {f} failed indices')
    
    edges = np.linspace(0,2500,51)
    plt.figure(figsize=(10,6))
    plt.hist(avg_array_track, density=True, color='c', edgecolor='k', bins=edges, linewidth=0.5, alpha=0.7, label='True Track')
    plt.hist(avg_array_shower, density=True, color='m', edgecolor='k', bins=edges, linewidth=0.5, alpha=0.7, label='True Shower')
    plt.title(f'ADC Peak histogram for {n} events, cutoff = {hits_cutoff}')
    plt.xlabel('W position in event')
    plt.ylabel('Average ADC density')
    plt.legend()
    plt.grid(False)
    plt.show()


def smooth_adc_plot_pdg(events, event_idx, adc_min, sigma):
    shower_mask = [-11, 11, 22]
    
    # Finding x/w hits and ADC values for a single event
    w_hits = events.reco_hits_w[event_idx]
    x_hits = events.reco_hits_x_w[event_idx]
    adc_values = events.reco_adcs_w[event_idx]

    # filtering track or shower from pdg code
    pdg = events.mc_pdg[event_idx]
    track_or_shower = None # initialising the track or shower variable
    line_colour = None
    
    if pdg in shower_mask:
        track_or_shower = 'Shower'
        line_colour = 'm'
    else: 
        track_or_shower = 'Track'
        line_colour = 'c'
    
    # Check and adjust adc_min
    if min(adc_values) > adc_min:
        adc_min = min(adc_values)

    # Apply ADC filter
    adc_mask = (adc_values >= adc_min)
    filtered_w_hits = w_hits[adc_mask]
    filtered_x_hits = x_hits[adc_mask]
    filtered_adc_values = adc_values[adc_mask]

    # Plot if there are sufficient hits after filtering
    if len(filtered_w_hits) == len(filtered_x_hits) and len(filtered_w_hits) > 5:
        plt.figure(figsize=(8, 6))
        
        # Smoothing and plotting the ADC trend over W
        sorted_indices = np.argsort(filtered_w_hits)
        sorted_w_hits = filtered_w_hits[sorted_indices]
        sorted_adc_values = filtered_adc_values[sorted_indices]
        
        # Apply Gaussian smoothing to the ADC values
        smoothed_adc_values = gaussian_filter1d(sorted_adc_values, sigma=sigma)

        # invert w hits if the event goes to the left
        w_vertex = events.reco_particle_vtx_w[event_idx]
        avg_w = sum(w_hits)/len(w_hits)
        if avg_w < w_vertex:
            sorted_w_hits = sorted_w_hits[::-1]
    
        # move and squash filtered w hits
        sorted_w_hits = [x - min(sorted_w_hits) for x in sorted_w_hits]
        sorted_w_hits = [x/max(sorted_w_hits) for x in sorted_w_hits]
        
        # Plot the smoothed line
        plt.plot(sorted_w_hits, smoothed_adc_values, color=line_colour, linewidth=1.5, label=f'Smoothed {track_or_shower} ADC')
        
        # Plot labels and title
        plt.title(f'Smooth ADC timeline for particle idx {event_idx} for ADCs in [{adc_min:.1f}, {max(adc_values):.1f}]')
        plt.ylabel('ADC Gaussian')
        plt.xlabel('Positon in event')
        plt.grid(True)
        plt.legend()
        plt.show()
    else: print('Less than 5 hits in event, not sufficient to plot.')


# functions that will get us the PDFs

def var_rms(events, start_idx, end_idx, hits_cutoff, show_plot=False): # var1
    if not isinstance(start_idx, int) or not isinstance(end_idx, int) or start_idx < 0 or end_idx < 0 or start_idx > end_idx:
        return print('invalid entry of indices')

    track_rms = []
    shower_rms = []
    # loop over all particles between start/end idx inclusive
    for i in range(start_idx, 1 + end_idx):
        try:
            rms_value, indicator = rms_pdg(events, i, hits_cutoff)
            # Proceed to add to the appropriate list based on the indicator
            if rms_value is not None:
                if indicator == 'T':
                    track_rms.append(rms_value)
                elif indicator == 'S':
                    shower_rms.append(rms_value)
        except TypeError:
            # This block will execute if rms_pdg(i) returns None, skipping the unpacking
            pass

    # Manually get the fraction
    n_t = len(track_rms)
    n_s = len(shower_rms)
    
    # Define bin edges
    bin_edges = np.arange(0, 31, 1)
    
    # Calculate fractional weights for each entry
    track_weights = np.ones_like(track_rms) / n_t if n_t > 0 else np.zeros_like(track_rms)
    shower_weights = np.ones_like(shower_rms) / n_s if n_s > 0 else np.zeros_like(shower_rms)

    # Calculate histogram bin heights (fractional values) and store them in arrays
    track_bin_heights, _ = np.histogram(track_rms, bins=bin_edges, weights=track_weights)
    shower_bin_heights, _ = np.histogram(shower_rms, bins=bin_edges, weights=shower_weights)

    # Plot the histogram with fractional bin heights
    plt.figure(figsize=(10, 6))
    plt.hist(track_rms, bins=bin_edges, weights=track_weights, color='c', edgecolor='k', linewidth=0.5, alpha=0.5, label='Tracks')
    plt.hist(shower_rms, bins=bin_edges, weights=shower_weights, color='m', edgecolor='k', linewidth=0.5, alpha=0.5, label='Showers')

    # Set plot details
    plt.title(f'Fractional Histogram of {n_t + n_s} RMS Values for events above {hits_cutoff} hits, {n_t} tracks and {n_s} showers.')
    plt.xlabel('RMS Value')
    plt.ylabel('Fraction of events')
    plt.xticks(np.arange(0, 30, 2))
    plt.xlim(0, 30)
    plt.legend()
    plt.grid(False)
    if show_plot:
        plt.show()

    return track_bin_heights, shower_bin_heights

def p_rms(training_data, testing_events, event_idx): # training_data ss a tuple of arrays: track_bin_heights, shower_bin_heights; testing events: an events class; event_idx as usual
    pdf_t, pdf_s = training_data
    bin_edges = np.arange(0, 31, 1)

    val = rms(testing_events, event_idx, 2)

    p_idx = np.searchsorted(bin_edges, val) - 1

    if p_idx < 0 or p_idx > max(bin_edges):
        return 0,0
    
    p_t = pdf_t[p_idx]
    p_s = pdf_s[p_idx]

    return p_t, p_s


def var_adc_mean(events, start_idx, end_idx, hits_cutoff, show_plot=False): # var3
    if not isinstance(start_idx, int) or not isinstance(end_idx, int) or start_idx < 0 or end_idx < 0 or start_idx > end_idx:
        return print('invalid entry of indices')

    avg_array_track = []
    avg_array_shower = []
    # loop over all idx
    for event_idx in range(start_idx, end_idx + 1):

        w_hits = events.reco_hits_w[event_idx]
        if len(w_hits) > hits_cutoff:
            
            adc_values = events.reco_adcs_w[event_idx]
            w_adc_avg = np.mean(adc_values)
        
            # filtering track or shower from pdg code
            pdg = events.mc_pdg[event_idx]
            shower = False # initialising the track or shower variable

            if pdg in [-11, 11, 22]:
                shower = True

            if shower:
                avg_array_shower.append(w_adc_avg)
            else: avg_array_track.append(w_adc_avg)

    # Manually get the fraction
    n_t = len(avg_array_track)
    n_s = len(avg_array_shower)
    
    # Define bin edges
    bin_edges = np.arange(0,2501,50)
    
    # Calculate fractional weights for each entry
    track_weights = np.ones_like(avg_array_track) / n_t if n_t > 0 else np.zeros_like(avg_array_track)
    shower_weights = np.ones_like(avg_array_shower) / n_s if n_s > 0 else np.zeros_like(avg_array_shower)

    # Calculate histogram bin heights (fractional values) and store them in arrays
    track_bin_heights, _ = np.histogram(avg_array_track, bins=bin_edges, weights=track_weights)
    shower_bin_heights, _ = np.histogram(avg_array_shower, bins=bin_edges, weights=shower_weights)

    # Plot the histogram with fractional bin heights
    plt.figure(figsize=(10, 6))
    plt.hist(avg_array_track, bins=bin_edges, weights=track_weights, color='c', edgecolor='k', linewidth=0.5, alpha=0.5, label='Tracks')
    plt.hist(avg_array_shower, bins=bin_edges, weights=shower_weights, color='m', edgecolor='k', linewidth=0.5, alpha=0.5, label='Showers')

    # Set plot details
    plt.title(f'Fractional Histogram of {n_t + n_s} ADC Mean for events above {hits_cutoff} hits, {n_t} tracks and {n_s} showers.')
    plt.xlabel('Mean ADC')
    plt.ylabel('Fraction of events')
    plt.xlim(0, 2500)
    plt.legend()
    plt.grid(False)
    if show_plot:
        plt.show()

    return track_bin_heights, shower_bin_heights

def p_adc_mean(training_data, testing_events, event_idx):
    pdf_t, pdf_s = training_data
    bin_edges = np.linspace(0,2500,51)

    adc_values = testing_events.reco_adcs_w[event_idx]
    w_adc_avg = np.mean(adc_values)

    p_idx = np.searchsorted(bin_edges, w_adc_avg) - 1

    p_t = pdf_t[p_idx]
    p_s = pdf_s[p_idx]

    return p_t, p_s


def adc_profile_idx(events, event_idx):
    
    # finding x/w hits again for a single event
    w_hits = events.reco_hits_w[event_idx]
    adc_values = events.reco_adcs_w[event_idx]

    if len(w_hits) > 5:
        plt.figure(figsize = (8,6))
        plt.scatter(w_hits, adc_values, c='g', s=3, alpha=0.9) # scatter plot
        plt.title(f'ADC View for particle idx {event_idx}')
        plt.ylabel('X (W View)')
        plt.xlabel('W (Wire Position)')
        plt.grid(True)
        plt.show()
    else: return print('Less that 5 hits in event, not sufficient to plot.')


def ROC(pdf): # takes a tuple of p_t, p_s as pdf
    p_t = pdf[0]
    p_s = pdf[1]
    eff = []
    pur = []

    for i in range(0, len(p_s)):
        
        shower_right = sum(p_s[i:])
        both_right = sum(p_s[i:]) + sum(p_t[i:])
        shower_all = sum(p_s)

        e = shower_right / shower_all
        
        # Check to avoid division by zero in purity calculation
        if both_right != 0:
            p = shower_right / both_right  # Purity
        else:
            p = 1  # or you could continue to the next iteration with "continue"

        eff.append(e)
        pur.append(p)

    plt.plot(eff, pur)
    plt.scatter(eff, pur, s=15, marker='x')
    plt.scatter(1, 1, c='orange', label="Ideal", s=15, marker='x')
    plt.xlabel("Efficiency")
    plt.ylabel("Purity")
    plt.title("Efficiency vs Purity Curve for picking a shower")
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.legend()
    plt.show()

def ROC_alt(pdf): # takes a tuple of p_t, p_s as pdf
    p_t = pdf[0]
    p_s = pdf[1]
    eff = []
    pur = []

    for i in range(0, len(p_t)):
        
        track_right = sum(p_t[i:])
        both_right = sum(p_s[i:]) + sum(p_t[i:])
        track_all = sum(p_t)

        e = track_right / track_all
        
        # Check to avoid division by zero in purity calculation
        if both_right != 0:
            p = track_right / both_right  # Purity
        else:
            p = 1  # or you could continue to the next iteration with "continue"

        eff.append(e)
        pur.append(p)

    plt.plot(eff, pur)
    plt.scatter(eff, pur, s=15, marker='x')
    plt.scatter(1, 1, c='orange', label="Ideal", s=15, marker='x')
    plt.xlabel("Efficiency")
    plt.ylabel("Purity")
    plt.title("Efficiency vs Purity Curve for picking a track")
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.legend()
    plt.show()


def feature_histogram_2(feature_name, e_array, gamma_array, bin_edges):

    n_t = len(e_array)
    n_s = len(gamma_array)

    # Calculate fractional weights for each entry
    e_weights = np.ones_like(e_array) / n_t if n_t > 0 else np.zeros_like(e_array)
    gamma_weights = np.ones_like(gamma_array) / n_s if n_s > 0 else np.zeros_like(gamma_array)

    # Calculate histogram bin heights (fractional values) and store them in arrays
    e_bin_heights, _ = np.histogram(e_array, bins=bin_edges, weights=e_weights)
    gamma_bin_heights, _ = np.histogram(gamma_array, bins=bin_edges, weights=gamma_weights)

    # Plot the histogram with fractional bin heights
    plt.figure(figsize=(10, 6))
    plt.hist(e_array, bins=bin_edges, weights=e_weights, color='limegreen', edgecolor='k', linewidth=0.5, alpha=0.5, label='e')
    plt.hist(gamma_array, bins=bin_edges, weights=gamma_weights, color='purple', edgecolor='k', linewidth=0.5, alpha=0.5, label='gamma')
    plt.xlabel(f'feature: {feature_name}')
    plt.ylabel('fraction of events')
    plt.legend()
    plt.show()

    return e_bin_heights, gamma_bin_heights, bin_edges


def line(events, event_idx):
    w_hits = np.array(events.reco_hits_w[event_idx])
    x_hits = np.array(events.reco_hits_x_w[event_idx])
    
    # Calculate differences between consecutive points
    dx = np.diff(w_hits)
    dy = np.diff(x_hits)
    
    # Compute segment lengths
    segment_lengths = np.sqrt(dx**2 + dy**2)
    
    # Total arc length (line integral)
    total_length = np.sum(segment_lengths)
    
    # Normalize by the number of points
    normalised_length = total_length / len(w_hits)

    return normalised_length


def q4_ratio(events, event_idx):
    adcs = events.reco_adcs_w[event_idx]

    q4_idx = len(adcs) // 4

    adcs_q4 = adcs[-q4_idx:]

    ratio = sum(adcs_q4) / sum(adcs)

    return ratio


from sklearn.cluster import DBSCAN
# from before 
def correlation(events, event_idx):
    x_hits = events.reco_hits_x_w[event_idx]
    w_hits = events.reco_hits_w[event_idx] 

    # Check if there are valid hits
    if len(w_hits) == len(x_hits) and len(w_hits) > 15: # talk about advantages and disadvantages of results with a threshold 
        if np.std(x_hits) == 0 or np.std(w_hits) == 0:
            return None  # No valid correlation if there's no variation in data
        
        correlation = np.corrcoef(x_hits, w_hits)[0, 1]
        
        # Fit line using w_hits for x and calculate predicted y-values
        line_fit = np.polyfit(w_hits, x_hits, 1)
        line_y_pred = np.polyval(line_fit, w_hits)
        
        # Calculate line error between predicted and actual x_hits
        line_error = np.mean((x_hits - line_y_pred) ** 2)
        
        # Normalize scores
        correlation_score = abs(correlation) if not np.isnan(correlation) else 0
        error_score = max(0, 1 - line_error / 20) if line_error < 20 else 0
        
        # Weighted score
        line_score = (correlation_score * 0.7) + (error_score * 0.3)
        
        return (line_score * 100)  # Return the score and category

    else:
        return None

# from before 

def noise(events, event_idx, eps=2, min_samples=5):
    # Extract hit positions (no PDG filtering, just use reco hits)
    x_hits = events.reco_hits_x_w[event_idx]
    w_hits = events.reco_hits_w[event_idx]

    # Check if there are valid hits
    if len(w_hits) == len(x_hits) and len(w_hits) > 15:
        # Combine the coordinates for clustering
        hits_coordinates = np.column_stack((w_hits, x_hits))

        # Apply DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(hits_coordinates)
        labels = db.labels_

        # Count noise points (labeled as -1)
        n_noise = np.sum(labels == -1)

        # Count clusters (unique labels excluding -1)
        unique_clusters = set(labels) - {-1}
        n_clusters = len(unique_clusters)

        return n_noise + n_clusters
    else:
        return None

def angle(events, event_idx):
    x_hits = events.reco_hits_x_w[event_idx]
    w_hits = events.reco_hits_w[event_idx]

    if len(w_hits) == len(x_hits) and len(w_hits) > 15:
        # Fit the best-fit line
        line_fit = np.polyfit(w_hits, x_hits, 1)
        line_slope = line_fit[0]
        line_intercept = line_fit[1]

        # Calculate residuals (distance from the line)
        line_y_pred = np.polyval(line_fit, w_hits)
        residuals = np.abs(x_hits - line_y_pred)

        # Find the index of the furthest point
        furthest_idx = np.argmax(residuals)
        furthest_point = np.array([x_hits[furthest_idx], w_hits[furthest_idx]])

        # Start of the line is at the minimum W-coordinate
        min_w = np.min(w_hits)
        start_point = np.array([line_slope * min_w + line_intercept, min_w])

        # End of the red line (best-fit line) at the maximum W-coordinate
        max_w = np.max(w_hits)
        end_of_red_line = np.array([line_slope * max_w + line_intercept, max_w])

        # Calculate the lengths of the three sides of the triangle
        red_line_length = np.linalg.norm(end_of_red_line - start_point)  # Distance between start and end of red line
        purple_line_length = np.linalg.norm(furthest_point - start_point)  # Distance between start and furthest point (purple line)
        third_line_length = np.linalg.norm(furthest_point - end_of_red_line)  # Distance between end of red line and furthest point (third line)

        # Using the cosine rule to calculate the angle between the red and purple lines
        cos_theta = (red_line_length**2 + purple_line_length**2 - third_line_length**2) / (2 * red_line_length * purple_line_length)
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip value to avoid out-of-bound errors
        angle_degrees = np.degrees(angle_radians)  # Convert radians to degrees
        
        return angle_degrees
    else:
        return None

def slice_events(events):
    n = len(events.mc_pdg)
    
    identifiers = events.event_number
    data = np.arange(0, len(events.event_number))
    slices = []
    start_idx = 0  # Starting index for the current slice

    for i in range(1, len(identifiers)):
        # If the identifier changes, slice the data
        if identifiers[i] != identifiers[i - 1]:
            slices.append(data[start_idx:i])
            start_idx = i  # Update start index for the next slice

    # Add the last slice
    slices.append(data[start_idx:])

    m = len(slices)
    print(f'Events sliced, {n} events split across {m} unique event ids')

    return slices

def identify_candidate(events):
    identifiers = events.event_number
    data = np.arange(0, len(events.event_number))
    slices = []
    start_idx = 0

    # Split data into slices based on changes in identifiers
    for i in range(1, len(identifiers)):
        if identifiers[i] != identifiers[i - 1]:
            slices.append(data[start_idx:i])
            start_idx = i  # Update start index for the next slice

    slices.append(data[start_idx:])

    results = []

    for event_number, event_indices in enumerate(slices):  # Enumerate slices to get the event number

        w_hits_event = []
        indices = []  # To keep track of the corresponding `i` values
        
        for i in event_indices:
            w_hits_event.append(events.reco_hits_w[i])
            indices.append(i)  # Store the corresponding `i` values
        
        # Find the index of the maximum length in w_hits_event
        max_idx = max(range(len(w_hits_event)), key=lambda idx: len(w_hits_event[idx]))
        
        # Retrieve the corresponding `i` value
        candidate_idx = indices[max_idx]

        results.append((event_number, candidate_idx))
    
    return results

import itertools

def plot_sliced_event(events, event_number):
    event_indices = slice_events(events)[event_number]

    w_hits_event = []
    x_hits_event = []

    for i in event_indices:
        w_hits_event.append(events.reco_hits_w[i])
        x_hits_event.append(events.reco_hits_x_w[i])

    w_candidate = max(w_hits_event, key=len)
    x_candidate = max(x_hits_event, key=len)

    w_flattened = list(itertools.chain(*w_hits_event))
    x_flattened = list(itertools.chain(*x_hits_event))

    w_vtx = events.neutrino_vtx_w[event_indices[0]]
    x_vtx = events.neutrino_vtx_x[event_indices[0]]
    
    if len(w_hits_event) < 3:
        print(f'Event Number {event_number} has less than 3 hits and may appear insignificant')
    plt.figure(figsize = (12,8))
    plt.scatter(w_flattened, x_flattened, c='grey', s=4, label='Hits')
    plt.scatter(w_candidate, x_candidate, c='g', s=5, label='Candidate Lepton')
    plt.scatter(w_vtx, x_vtx, marker='*', facecolors='none', edgecolors='b', s=50, label='Neutrino Vertex')
    plt.title(f'W View for Event Number: {event_number}, with highlighted candidate lepton')
    plt.ylabel('X')
    plt.xlabel('W')
    plt.legend()
    plt.grid(True)
    plt.show()