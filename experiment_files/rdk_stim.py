# imports
from __future__ import division
import numpy as np
from psychopy import visual


# ------------------------ Define and initialize class and set global parameters ------------------------

# All dots are divided into 3 interleaved groups. On each frame, only 1 group is drawn on screen;
# the other 2 groups are absent from the screen on that frame but will be drawn when their turn comes in the round-robin sequence.
# Each group is drawn once every 3 frames: on its active frame the dot appears,
# then it is absent for 2 frames, then reappears at a new position — producing apparent motion (Movshon/Newsome algorithm).
     

class RDK:
    """
    RDK class constructor.
    Sets up all trial-invariant properties of the dot field:
    - stores basic stimulus parameters (density, speed, frame rate, field diameter, number of sequences),
    - computes the circular field area and total number of dots (n_dots),
    - ensures n_dots is divisible by n_sequences and derives dots per sequence,
    - configures dot lifetime tracking,
    - creates the PsychoPy ElementArrayStim.

    """   

    # Parameters: 
    def __init__(
            self, 
            win,                        # window where dots will be drawn in the single_trial class
            dot_density=12.0,           # [dots/deg²/s] chosen to match both the instantaneous density (0.20 dots/deg²/frame)
                                        # and dot count (~50 dots/frame) from Pilly & Seitz (2009)
                                        # Their setup: 0.20 dots/deg²/frame at 85 Hz → ~50 dots/frame
                                        # At 120 Hz, to preserve the same instantaneous density and dot count:
                                        # 0.20 dots/deg²/frame × 120 Hz = 24.0 dots/deg²/s → ~51 dots/frame
            dot_speed = 12,             # [deg/s] default; overridden by Experiment based on target_displacement
            frame_rate = 60,           # [Hz] = [frame/s]
            field_diameter = 18.0,      # [deg] diameter of the circular aperture where dots are sampled and respawned;
                                        # passed as the fieldSize parameter to PsychoPy's ElementArrayStim 
            n_sequences=3,              # total number of interleaved sequences
            rng= None,                  # local random generator; if None, a new one is created
            max_lifetime_frames = 36    # how many video frames a dot stays alive (counted every frame, not just when active);
                                        # since each dot appears once every 3 frames: 36 frames / 3 = 12 appearances before dying.
    ):
        
     

        # ----- Basic stimulus parameters -----

        self.dot_speed = dot_speed
        self.frame_rate = frame_rate
        self.field_diameter = field_diameter
        self.dot_density = dot_density
        self.n_sequences = n_sequences
        self.field_radius = field_diameter/2.0      # [deg] radius of the circular aperture (diameter divided by 2)
 
        # ----- Random generator -----

        # Local random generator for reproducibility 
        self.rng = rng or np.random.default_rng()


        # ----- Total number of dots -----

        # Calculate the area of the circular field where dots are sampled and respawned; area of a circle = π * radius^2  [deg²].
        # The aperture is circular to avoid directional bias that can occur if the edges of a square are also filled with dots.
        field_area = np.pi * self.field_radius**2

        # Calculate the total number of dots in the pool 
        self.n_dots = int(np.ceil(self.dot_density * field_area/self.frame_rate)) * self.n_sequences
            # n_dots - total dot pool the algorithm manages internally across all n_sequences groups;
            # only n_dots_in_sequence dots (= n_dots / n_sequences) are drawn on any given frame;
            # the remaining dots are drawn when their sequence is active
            # Step 1: dot_density * field_area / frame_rate = n_dots_in_sequence
            #         — how many dots are visible on screen on any single frame
            #         — Units: [dots/deg²/s] * [deg²] / [Hz] = [dots]
            # Step 2: × n_sequences = n_dots
            #         — multiply by 3 because all 3 groups must be stored in memory simultaneously;
            #           each group needs to remember its positions between its turns    
        
        # Add extra dots if necessary to make the total number divisible by n_sequences 
        self.n_dots += (self.n_sequences - self.n_dots % self.n_sequences) % self.n_sequences
            # More explicit version of the same logic: 
            # dots_remainder = self.n_dots % self.n_sequences                        (what is left after dividing by n_sequences)
            # extra_dots = (self.n_sequences - dots_remainder) % self.n_sequences    (how many dots to add)
            # self.n_dots += extra_dots                                              (add the extra dots) 


        # Calculate the number of dots in each sequence (group).
        # All 3 groups have this same size. On any frame, exactly one group
        # (n_dots_in_sequence dots) is drawn on screen.
        self.n_dots_in_sequence = self.n_dots // self.n_sequences
        if (self.n_dots_in_sequence * self.n_sequences != self.n_dots):
            raise ValueError("Inconsistent dot counts: n_dots_in_sequence * n_sequences "
                             "must equal n_dots (computed from dot_density, frame_rate and field_diameter).")
            # Sanity check: we should be able to reconstruct the total number of dots
            # from n_dots_in_sequence * n_sequences. If not, something is inconsistent
            # with how n_dots was computed from dot_density, frame_rate and field_diameter.
        

        # Pre-compute for logging — available before any trial starts.
        self.spatial_displacement = float(self.dot_speed / self.frame_rate * self.n_sequences)  # how far a dot jumps each time it appears [deg]
        self.temporal_displacement = float(self.n_sequences / self.frame_rate * 1000.0)         # time between consecutive appearances of the same dot [ms]
        self.instantaneous_dot_density = float(self.dot_density / self.frame_rate)              # dot density per frame [dots/deg²/frame]

        
        # ----- Lifetime tracking -----

        # Lifetime tracking: how long each dot has been alive (in frames) and how many expired on the last frame. 
        # Each dot, regardless of whether it is in the active sequence or not, has a
        # maximum age of max_lifetime_frames video frames. On every frame, all dot_lifetimes are incremented by 1.
        # When a dot’s lifetime reaches max_lifetime_frames (or it leaves the aperture), it is respawned at a new random
        # location inside the circular field and its lifetime is reset to 0.
        # Note: dots belong to 1 of 3 interleaved sequences, so their motion is updated only every 3rd frame, but lifetime
        # is counted on EVERY frame.
        self.max_lifetime_frames = int(max_lifetime_frames)   # max age before a dot is respawned
        self.dot_lifetimes = np.zeros(self.n_dots, dtype=int) # current age of each dot (in frames)
        self.n_expired_last = 0                               # diagnostic: number of dots that expired on the previous frame
        

        # ----- ElementArrayStim creation -----

        # Create the ElementArrayStim that draws exactly n_dots_in_sequence dots per frame.
        self.dots_stim = visual.ElementArrayStim(
            win, 
            elementTex=None,            # no texture: solid dots, shape defined only by elementMask
            fieldShape='circle',        # documents that this stimulus is intended as a circular field,
                                        # but the actual sampling from the the circle are handled manually in the code.
            elementMask='circle',       # render each dot as a circle
            sizes=0.08,                 # size of each dot in degrees of visual angle
            nElements = self.n_dots_in_sequence,    # the number of dots in one group (1/3 of all dots), drawn per frame
            units='deg', 
            fieldSize = self.field_diameter,
            colors=[1, 1, 1],           # dot color (reference: [1,1,1] is max white)
            colorSpace='rgb',           # defines what the numbers mean
            )  


# ------------------------------ Initialize stimulus -----------------------------
    

    def initialize_rdk_stim(self, direction, coherence):

        """
        This method prepares the dot field for a new trial.
        It:
        - sets trial parameters: motion direction [deg] and coherence [0-1],
        - computes dot displacement per frame and per update (because only one 
        of the n_sequences groups moves on a given frame),
        - converts direction to radians and defines X/Y movement for signal dots,
        - creates `active_dots_mask` for the whole dot pool,
        - samples initial dot positions inside the circular aperture [deg],
        - resets the current sequence index and dot lifetimes [frames] 
        """  
        
        # ----- Trial-specific parameters -----

        # Set trial-specific parameters: motion direction [deg] and coherence in [0, 1]
        self.direction = float(direction)  
        self.coherence = float(coherence)
        
        # ----- Global motion direction and displacement (per frame and per sequence) -----

        # Compute how far a dot would move per video frame if it were updated on every frame
        displacement_per_frame = self.dot_speed/self.frame_rate  #[deg/frame]
        
        # Compute how far a dot actually moves when its sequence is updated 
        # (each group is updated once every 3 frames)
        displacement_per_sequence = displacement_per_frame * self.n_sequences #[deg/update]
            # If we did NOT multiply by n_sequences, dots would move only 1/3 of the intended distance
            # (for n_sequences = 3), so their effective speed would be three times too slow. Multiplying
            # by n_sequences ensures that, even though each group moves only every 3rd frame, the overall
            # speed still matches dot_speed.
            # Example sanity check:
            #   dot_speed    = 12 deg/s
            #   frame_rate   = 120 Hz  (≈ 1/120 s per frame)
            #   n_sequences  = 3
            #   displacement_per_frame   = 12 / 120 ≈ 0.1 deg/frame
            #   displacement_per_update  = 0.1 * 3 ≈ 0.3 deg/update
            # So:
            #   - If a dot moved every frame: 0.1 deg/frame → 12 deg/s overall.
            #   - Here it moves only every 3rd frame, but by 0.3 deg each time → still 12 deg/s overall.

        
        # Compute global motion direction for this trial.
        # theta: single angle [rad] that sets the coherent motion direction for all signal dots 
        # (0° = right, 180° = left). Conversion from degrees to radians is necessary because NumPy’s
        # trig functions (np.cos / np.sin) expect radians, not degrees.
        theta = np.deg2rad(self.direction)

        # Compute one displacement vector for signal dots in the active sequence.
        self.displacement_vector_X = displacement_per_sequence * np.cos(theta)
        self.displacement_vector_Y = displacement_per_sequence * np.sin(theta) 
            # Displacement vectors specify how far a signal (coherent) dot moves along X and Y on each update [deg/update].
            # Noise dots are also part of the active sequence, but they ignore this vector and are later
            # repositioned randomly in update_rdk_stim().
            # In this task (0° = right, 180° = left) the Y component is zero, but we keep both
            # components for clarity and for possible future motion directions. 


        # ----- Active dots mask for the current sequence -----

        # Create a boolean mask of length n_dots to track which dots belong to the group being updated in the current sequence 
        # (i.e. dots active on this frame). Initially all values are False; later, in update_rdk_stim, on each frame, 
        # 1 out of n_sequences subsets is set to True, giving exactly n_dots_in_sequence True entries (because n_dots is divisible by n_sequences).  
        self.active_dots_mask = np.zeros(self.n_dots, dtype=bool)   
                                                              
        
        # ----- Initial dot positions inside circular aperture -----
  
        # Set initial dot positions – dots live inside a circle with a given radius [deg].
        radius = self.field_radius

        # Create an array for all dots and assign each dot one random float that represents an orientation angle 
        # (the direction of the dot from the centre of the circle along a line). 
        # rng.random() gives numbers in [0, 1); multiplying by 2π converts this to angles in [0, 2π) radians 
        rand_theta = self.rng.random(self.n_dots) * 2.0 * np.pi

        # Create random distances for each dot: rng.random(self.n_dots) gives values in [0, 1).
        # Taking sqrt(...) makes values less clustered near 0, so dots are spread across the whole circle.
        # Multiplying by radius gives the final distance from the centre, as a value between 0 and radius [deg].
        rand_r = np.sqrt(self.rng.random(self.n_dots)) * radius 
        
        # Convert polar coordinates (distance from centre, angle) into Cartesian coordinates (x, y) in degrees.
        # x is how far the dot is from the centre horizontally (left/right),
        # y is how far the dot is from the centre vertically (up/down).
        x = rand_r * np.cos(rand_theta)
        y = rand_r * np.sin(rand_theta)

        # Store all dot positions in one array with shape (2, n_dots):
        # - row 0: X coordinates of all dots
        # - row 1: Y coordinates of all dots
        self.dots_coordinates = np.vstack([x, y])    
        
        # ----- Sequence index and lifetimes reset -----
        
        # Current sequence of dots being updated (0, 1, or 2).
        # Start at -1 so that the first call to update_rdk_stim() will advance it to 0.
        self.current_sequence_index = -1  

        
        # Reset lifetimes and diagnostics at the start of each trial.
        self.dot_lifetimes[:] = self.rng.integers(low=0, high=self.max_lifetime_frames, size=self.n_dots, dtype=int)  # one integer per dot: initial dot lifetimes are randomly 
                                                                                                                      # sampled between 0 and the maximum lifetime so that dot expiration and respawning 
                                                                                                                      # are spread out over time rather than synchronized
        self.n_outside_last = 0     # number of dots respawned for leaving the aperture on the last update
        self.n_expired_last = 0     # number of dots respawned for exceeding max_lifetime_frames on the last update

        
        return self.active_dots_mask, self.dots_coordinates, self.current_sequence_index


# ------------------------ Update stimulus ------------------------

    def update_rdk_stim(self):
        """
        This method advances dots by one frame. 
        On each call, the method:
        - increases the age of all dots;
        - selects which group (sequence) of dots is active on this frame, moving from one group to the next each frame
        - for the active dots, randomly assigns signal or noise status according to the current coherence level;
        - moves signal dots coherently in the specified motion direction;
        - repositions noise dots to new random locations within the aperture;
        - handles dots that leave the aperture by placing them back inside;
        - handles dots whose lifetime has expired by assigning them new random positions 
        """

        # ----- Dot lifetime update and active sequence selection -----

        # Increase age of all dots each frame
        self.dot_lifetimes += 1
        
        # Each frame, advance the sequence index by 1 and wrap it cyclically through 0, 1, ..., n_sequences-1 using modulo
        self.current_sequence_index = (self.current_sequence_index + 1) % self.n_sequences

        # Set all currently active dots in active_dots_mask to inactive (False) before activating the next group of dots  
        # (this prevents accumulation of active dots from previous sequences and maintains the logic that only one group is active at a time) 
        self.active_dots_mask[self.active_dots_mask] = False

        # Mark all dots in the current sequence as active (True)
        self.active_dots_mask[self.current_sequence_index::self.n_sequences] = True    
            # Using the slice [current_sequence_index : end : n_sequences] selects all dots in this sequence
            # (e.g., for current_sequence_index = 0 and n_sequences = 3 -> indices 0, 3, 6, ...).
            # This always selects exactly n_dots_in_sequence dots (= n_dots / n_sequences).

        
        # ----- Active dots count -----

        # Count how many dots are active in the current sequence by counting True values in active_dots_mask
        current_dots_count = int(np.sum(self.active_dots_mask))     
        self.current_dots_count = current_dots_count   

        # n_dots_in_sequence was computed earlier by dividing the total number of dots
        # in the stimulus (self.n_dots) by the number of sequences (self.n_sequences).
        # In principle, current_dots_count should be equal to n_dots_in_sequence.
        # Here we recompute it from active_dots_mask (counting True values) as a safer
        # and more flexible consistency check.
        if current_dots_count != self.n_dots_in_sequence:
            raise RuntimeError(
                f"Active dots mismatch: got {current_dots_count}, "
                f"expected {self.n_dots_in_sequence}"
            )                                        


        # ----- Signal and noise mask assignment -----

        # Create a boolean mask for all dots (length equal to the total number of dots).
        # Initially all values are False; on each frame, signal dots are marked True (only out of active dots). 
        self.signal_dots_mask = np.zeros((self.n_dots), dtype=bool)  
   
        # Decide which active dots are signal and which are noise: for each active dot, draw a random float in [0, 1) and mark it as signal (True)
        # if the value is smaller than coherence; otherwise mark it as noise (False).
        current_signal_mask = self.rng.random(current_dots_count) < self.coherence  
    
        # Compute the number of signal dots by counting True values in current_signal_mask
        n_signal_dots = int(np.sum(current_signal_mask))

        # Compute the number of noise dots as the difference between current_dots_count and n_signal_dots
        n_noise_dots = current_dots_count - n_signal_dots 

        self.n_signal_dots = n_signal_dots      
        self.n_noise_dots = n_noise_dots        

        # In the signal mask for all dots (signal_dots_mask), assign True/False only to dots
        # that are currently active (where active_dots_mask is True), using values from
        # current_signal_mask (a boolean array with one value per active dot: True = signal, False = noise).
        # Dots that are not active remain False in signal_dots_mask.
        self.signal_dots_mask[self.active_dots_mask] = current_signal_mask
 
        # Create the noise mask: it marks which active dots are treated as noise on this frame.
        # Start by copying the active mask, so all active dots are temporarily marked as candidates for noise.
        self.noise_dots_mask = self.active_dots_mask.copy()

        # In the noise mask, set all dots that are signal on this frame (where signal_dots_mask is True) to False, 
        # so only noise dots remain True.
        self.noise_dots_mask[self.signal_dots_mask] = False  
        
 
        # ----- Signal dot motion and noise dot resampling -----

        # Update coordinates of signal dots by adding the X and Y components of the displacement vector to their current positions
        self.dots_coordinates[0, self.signal_dots_mask] += self.displacement_vector_X
        self.dots_coordinates[1, self.signal_dots_mask] += self.displacement_vector_Y
        
        # For each noise dot, pick a random angle (0–2π) and a random radius (0–field_radius),
        # then convert these polar coordinates (r, θ) to Cartesian (x, y).
        # As a result, noise dots flicker by jumping to new random locations on each frame
        if n_noise_dots > 0:
            # Same sampling scheme as in initialize_rdk_stim: random angle + radius (uniform in circle)
            radius = self.field_radius

            rand_theta = self.rng.random(n_noise_dots) * 2.0 * np.pi

            rand_r = np.sqrt(self.rng.random(n_noise_dots)) * radius
            
            new_x = rand_r * np.cos(rand_theta) # [deg]
            new_y = rand_r* np.sin(rand_theta)  # [deg]

            self.dots_coordinates[0, self.noise_dots_mask] = new_x
            self.dots_coordinates[1, self.noise_dots_mask] = new_y 


        # ----- Outside and expired dot respawning -----

        # After updating signal and noise positions, fix any dots that are outside the aperture
        # or have exceeded their lifetime by respawning them inside the circle and resetting lifetime.
        # Edge handling: some implementations use separate approaches for signal and noise dots, 
        # but this was deemed unnecessary here, because dots rarely leave the circular aperture with the current parameters.
        x_all = self.dots_coordinates[0, :]   # [deg]
        y_all = self.dots_coordinates[1, :]   # [deg]

        # Compute squared distance from the center for each dot: r^2 = x^2 + y^2
        r2_all = x_all**2 + y_all**2          # [deg^2]
        outside_dots_mask = r2_all > (self.field_radius ** 2)

        # Count how many dots are currently outside the aperture
        n_outside = int(np.sum(outside_dots_mask))
        
        # Detect dots whose lifetime has reached or exceeded the maximum
        expired_mask = self.dot_lifetimes >= self.max_lifetime_frames
        n_expired = int(np.sum(expired_mask))

        self.n_outside_last = n_outside
        self.n_expired_last = n_expired        

        # Create a mask of active dots that need to be respawned (either outside the aperture or lifetime-expired)
        respawn_mask = (outside_dots_mask | expired_mask) & self.active_dots_mask
        n_respawn = int(np.sum(respawn_mask))

        if n_respawn > 0:
            # Sample new positions inside the circular aperture (same scheme as initialization)
            radius = self.field_radius
            rand_theta = self.rng.random(n_respawn) * 2.0 * np.pi
            rand_r = np.sqrt(self.rng.random(n_respawn)) * radius

            new_x = rand_r * np.cos(rand_theta)   # deg
            new_y = rand_r * np.sin(rand_theta)   # deg

            # Assign new positions and reset lifetime for all respawned dots
            self.dots_coordinates[0, respawn_mask] = new_x
            self.dots_coordinates[1, respawn_mask] = new_y
            self.dot_lifetimes[respawn_mask] = 0


        # ----- ElementArrayStim coordinates update -----
        
        # Prepare coordinates for drawing: dots_coordinates has shape (2, n_dots) → [0, :] = x, [1, :] = y.
        # active_dots_mask has exactly n_dots_in_sequence True entries; 
        # indexing with active_dots_mask selects the n_dots_in_sequence columns of the current
        # sequence. ElementArrayStim expects positions as (n_dots_in_sequence, 2), so we transpose.
        xys = self.dots_coordinates[:, self.active_dots_mask].T  # shape: (n_dots_in_sequence, 2)

        # xys is passed to the ElementArrayStim (dots_stim) to update dot positions in the stimulus
        self.dots_stim.setXYs(xys)
                        
        return self.active_dots_mask, self.dots_coordinates, self.current_sequence_index

    
# ---------------------------------------------------------

# --------- DIAGNOSTICS / STATS METHODS (to be removed before the pilot study) ---------

    def get_frame_snapshot(self, frame_count):
        
        """
        Collect a diagnostic snapshot of the current RDK state for this frame.

        Returns
        -------
        snapshot : dict
            A dictionary of scalar values (ints/floats) describing:
            - which sequence is active,
            - how many dots are active / signal / noise / respawned,
            - empirical coherence and mask fractions,
            - basic radial statistics for dot positions,
            - trial-level motion parameters (direction, coherence),
            - optional x,y positions for the first N dots.
        """

        # ----- Basic counts and masks -----

        current_sequence_index = int(self.current_sequence_index)

        n_dots = int(self.n_dots)
        n_dots_in_sequence = int(self.n_dots_in_sequence)

        # Count how many dots are active, signal and noise on this frame.
        # If a mask hasn't been created yet (e.g. before first update), fall back to 0.
        active_count = int(self.active_dots_mask.sum()) if hasattr(self, "active_dots_mask") else 0
        n_signal_dots = int(self.signal_dots_mask.sum()) if hasattr(self, "signal_dots_mask") else 0
        n_noise_dots = int(self.noise_dots_mask.sum()) if hasattr(self, "noise_dots_mask") else 0

        # Values that are explicitly stored during update_rdk_stim (if available).
        # If they don't exist yet, we fall back to reasonable defaults.
        current_dots_count = int(getattr(self, "current_dots_count", active_count))
        n_outside_last = int(getattr(self, "n_outside_last", 0))
        n_expired_last = int(getattr(self, "n_expired_last", 0))


        # ----- Empirical coherence and fractions -----


        # Empirical coherence = signal dots / active dots.
        if active_count > 0:
            empirical_coherence = n_signal_dots / active_count
        else:
            empirical_coherence = float("nan")

        # Fractions of dots relative to the entire pool (useful for sanity checks).
        active_fraction = (active_count / n_dots) if n_dots > 0 else float("nan")
        signal_fraction_all = (n_signal_dots / n_dots) if n_dots > 0 else float("nan")
        noise_fraction_all  = (n_noise_dots / n_dots) if n_dots > 0 else float("nan")

        # ----- Radial summaries for dot positions -----

        if hasattr(self, "dots_coordinates"):
            x = self.dots_coordinates[0, :]
            y = self.dots_coordinates[1, :]
            r = np.sqrt(x**2 + y**2)        # distance from centre for each dot [deg]
            r_min = float(r.min())
            r_max = float(r.max())
            r_mean = float(r.mean())
        else:
            r_min = r_max = r_mean = float("nan")

        # ----- Displacement information for this trial -----

        # Displacement components are set in initialize_rdk_stim() and define
        # the coherent step per update for signal dots.
        disp_x = float(getattr(self, "displacement_vector_X", float("nan")))
        disp_y = float(getattr(self, "displacement_vector_Y", float("nan")))
        
        # Some code may precompute displacement_abs; otherwise compute hypot(disp_x, disp_y).
        if hasattr(self, "displacement_abs"):
            disp_abs = float(self.displacement_abs)
        else:
            disp_abs = float(np.hypot(disp_x, disp_y)) if np.isfinite(disp_x) and np.isfinite(disp_y) else float("nan")

        # ----- Trial-level parameters -----

        # Direction [deg] and coherence [0–1] are trial-specific values stored by initialize_rdk_stim().
        direction = float(getattr(self, "direction", float("nan")))
        coherence = float(getattr(self, "coherence", float("nan")))

        # Core frame-level diagnostic values for this RDK update.
        snapshot = {
            "frame_count": int(frame_count),
            "current_sequence_index": current_sequence_index,

            "n_dots": n_dots,
            "n_dots_in_sequence": n_dots_in_sequence,
            "current_dots_count": current_dots_count,

            "n_signal_dots": n_signal_dots,
            "n_noise_dots": n_noise_dots,
            "n_outside_last": n_outside_last,
            "n_expired_last": n_expired_last,  


            "empirical_coherence": float(empirical_coherence),
            "active_fraction": float(active_fraction),
            "signal_fraction_all": float(signal_fraction_all),
            "noise_fraction_all": float(noise_fraction_all),

            "r_min": r_min,
            "r_max": r_max,
            "r_mean": r_mean,

            "displacement_x": disp_x,
            "displacement_y": disp_y,
            "displacement_abs": disp_abs,

            "direction": direction,
            "coherence": coherence,
        }

        # Store positions for the first N dots to enable later trajectory plots / diagnostics
        N = 10
        if hasattr(self, "dots_coordinates"):
            
            # For existing dots, store x/y for up to N dots
            for idx in range(min(N, n_dots)):
                snapshot[f"x_dot{idx}"] = float(self.dots_coordinates[0, idx])
                snapshot[f"y_dot{idx}"] = float(self.dots_coordinates[1, idx])
            
            # If the total number of dots is smaller than N, add placeholder NaNs
            # so that x_dot0..x_dotN-1 always exist in the snapshot.
            for idx in range(n_dots, N):
                snapshot[f"x_dot{idx}"] = float("nan")
                snapshot[f"y_dot{idx}"] = float("nan")
        else:
            for idx in range(N):
                snapshot[f"x_dot{idx}"] = float("nan")
                snapshot[f"y_dot{idx}"] = float("nan")
        return snapshot