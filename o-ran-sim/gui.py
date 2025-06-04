import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
import math

# Import simulation components
from sim_core.params import SimParams
from sim_core.simulation import Simulation
# Ensure TF_AVAILABLE_GLOBAL reflects the true TensorFlow availability from dqn.py
from sim_core.constants import TF_AVAILABLE as TF_AVAILABLE_GLOBAL

# Import utilities
from utils.logger import LogHandler
# from utils.saver import SaveHandler # Not used directly by GUI, so can be commented out if not explicitly called


class CellularSimApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("O-RAN Sim :: RL Traffic Steering")
        self.geometry("1500x1000")

        self.params = SimParams()
        self.simulation = None
        self.param_widgets = {}
        self.all_runs_metrics = {}  # To store metrics from different agent runs for comparison

        self.log_handler = LogHandler()  # Initialize the log handler
        self._create_widgets()
        self._populate_params_from_simparams()
        self._toggle_ppp_params_visibility()
        # Bind the log_text widget to the log_handler
        self.log_handler.set_text_widget(self.log_text)

    def _create_widgets(self):
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Parameter Pane (Left)
        param_pane_container = ttk.Labelframe(
            self.main_frame, text="Simulation Parameters", padding="10"
        )
        param_pane_container.grid(row=0, column=0, sticky="ns", rowspan=3, padx=5)

        param_canvas = tk.Canvas(param_pane_container, width=380)
        param_scrollbar = ttk.Scrollbar(
            param_pane_container, orient="vertical", command=param_canvas.yview
        )
        self.scrollable_frame_params = ttk.Frame(param_canvas)
        self.scrollable_frame_params.bind(
            "<Configure>",
            lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all")),
        )
        param_canvas.create_window((0, 0), window=self.scrollable_frame_params, anchor="nw")
        param_canvas.configure(yscrollcommand=param_scrollbar.set)
        param_canvas.pack(side="left", fill="both", expand=True)
        param_scrollbar.pack(side="right", fill="y")

        current_row = 0
        # RL Agent Selection
        ttk.Label(self.scrollable_frame_params, text="RL Agent Type:").grid(
            row=current_row, column=0, sticky="w", padx=5, pady=3
        )
        self.param_widgets["rl_agent_type_var"] = tk.StringVar(
            value=self.params.rl_agent_type
        )

     
        agent_options = [
            "Baseline",
            "TabularQLearning",
            "SARSA",
            "ExpectedSARSA",
            "NStepSARSA",
            "DQN"
        ]

        # Conditionally remove DQN if TensorFlow is not available
        if not TF_AVAILABLE_GLOBAL:
            if "DQN" in agent_options:
                agent_options.remove("DQN")
                self.log_handler.log("Warning: TensorFlow not found. DQN agent option is disabled.")


        agent_menu = ttk.OptionMenu(
            self.scrollable_frame_params,
            self.param_widgets["rl_agent_type_var"],
            self.params.rl_agent_type,
            *agent_options,
        )
        agent_menu.grid(row=current_row, column=1, sticky="ew", padx=5, pady=3)
        current_row += 1

        # Placement Method
        ttk.Label(self.scrollable_frame_params, text="Placement Method:").grid(
            row=current_row, column=0, sticky="w", padx=5, pady=2
        )
        self.param_widgets["placement_method_var"] = tk.StringVar(
            value=self.params.placement_method
        )
        rb_uniform = ttk.Radiobutton(
            self.scrollable_frame_params,
            text="Uniform Random",
            variable=self.param_widgets["placement_method_var"],
            value="Uniform Random",
            command=self._toggle_ppp_params_visibility,
        )
        rb_uniform.grid(row=current_row, column=1, columnspan=1, sticky="w")
        current_row += 1
        rb_ppp = ttk.Radiobutton(
            self.scrollable_frame_params,
            text="PPP",
            variable=self.param_widgets["placement_method_var"],
            value="PPP",
            command=self._toggle_ppp_params_visibility,
        )
        rb_ppp.grid(row=current_row, column=1, columnspan=1, sticky="w")
        current_row += 1

        # Standard Params (num_bs, num_ues are for Uniform)
        self.param_widgets["num_bss_label"] = ttk.Label(
            self.scrollable_frame_params, text="Num BSs (Uniform):"
        )
        self.param_widgets["num_bss_label"].grid(
            row=current_row, column=0, sticky="w", padx=5, pady=2
        )
        self.param_widgets["num_bss"] = ttk.Entry(
            self.scrollable_frame_params, width=12
        )
        self.param_widgets["num_bss"].grid(
            row=current_row, column=1, sticky="ew", padx=5, pady=2
        )
        current_row += 1
        self.param_widgets["num_ues_label"] = ttk.Label(
            self.scrollable_frame_params, text="Num UEs (Uniform):"
        )
        self.param_widgets["num_ues_label"].grid(
            row=current_row, column=0, sticky="w", padx=5, pady=2
        )
        self.param_widgets["num_ues"] = ttk.Entry(
            self.scrollable_frame_params, width=12
        )
        self.param_widgets["num_ues"].grid(
            row=current_row, column=1, sticky="ew", padx=5, pady=2
        )
        current_row += 1

        # PPP Params (lambda_bs, lambda_ue)
        self.param_widgets["lambda_bs_label"] = ttk.Label(
            self.scrollable_frame_params, text="Lambda BS (per km²):"
        )
        self.param_widgets["lambda_bs_label"].grid(
            row=current_row, column=0, sticky="w", padx=5, pady=2
        )
        self.param_widgets["lambda_bs"] = ttk.Entry(
            self.scrollable_frame_params, width=12
        )
        self.param_widgets["lambda_bs"].grid(
            row=current_row, column=1, sticky="ew", padx=5, pady=2
        )
        current_row += 1
        self.param_widgets["lambda_ue_label"] = ttk.Label(
            self.scrollable_frame_params, text="Lambda UE (per km²):"
        )
        self.param_widgets["lambda_ue_label"].grid(
            row=current_row, column=0, sticky="w", padx=5, pady=2
        )
        self.param_widgets["lambda_ue"] = ttk.Entry(
            self.scrollable_frame_params, width=12
        )
        self.param_widgets["lambda_ue"].grid(
            row=current_row, column=1, sticky="ew", padx=5, pady=2
        )
        current_row += 1

        param_fields_common = [
            ("sim_area_x", "Sim Area X (m):"),
            ("sim_area_y", "Sim Area Y (m):"),
            ("time_step_duration", "Time Step (s):"),
            ("total_sim_steps", "Total Steps:"),
            ("bs_tx_power_dbm", "BS Tx Power (dBm):"),
            ("bs_link_beam_gain_db", "BS Link Gain (dB):"),
            ("bs_access_beam_gain_db", "BS Access Gain (dB):"),
            ("ue_speed_mps", "UE Speed (m/s):"),
            ("ue_noise_figure_db", "UE Noise Fig (dB):"),
            ("target_ue_throughput_mbps", "Target UE Rate (Mbps):"),
            ("max_rbs_per_ue", "Max RBs/UE (Total):"),
            ("max_rbs_per_ue_per_bs", "Max RBs/UE/BS (Dual):"),
            ("num_total_rbs", "Total RBs:"),
            ("rb_bandwidth_mhz", "RB BW (MHz):"),
            ("path_loss_exponent", "Path Loss Exp:"),
            ("ref_dist_m", "Ref Dist (m):"),
            ("ref_loss_db", "Ref Loss (dB):"),
            ("shadowing_std_dev_db", "Shadow StdDev (dB):"),
            ("ho_hysteresis_db", "HO Hyst (dB, Baseline):"),
            ("ho_time_to_trigger_s", "HO TTT (s, Baseline):"),
            ("min_rsrp_for_acq_dbm", "Min Acq RSRP (dBm):"),
            # RL Params
            ("rl_gamma", "RL Gamma:"),
            ("rl_learning_rate", "RL Learning Rate:"),
            ("rl_epsilon_start", "RL Eps Start:"),
            ("rl_epsilon_end", "RL Eps End:"),
            ("rl_epsilon_decay_steps", "RL Eps Decay Steps:"),
            ("rl_batch_size", "RL Batch Size (DQL):"),
            ("rl_target_update_freq", "RL Target Update (DQL):"),
            ("rl_replay_buffer_size", "RL Replay Buf (DQL):"),
            ("rl_n_step_sarsa", "RL N-step (SARSA):"),
        ]
        for name, label_text in param_fields_common:
            self.param_widgets[name + "_label"] = ttk.Label(
                self.scrollable_frame_params, text=label_text
            )
            self.param_widgets[name + "_label"].grid(
                row=current_row, column=0, sticky="w", padx=5, pady=2
            )
            self.param_widgets[name] = ttk.Entry(
                self.scrollable_frame_params, width=12
            )
            self.param_widgets[name].grid(
                row=current_row, column=1, sticky="ew", padx=5, pady=2
            )
            current_row += 1

        # Right side (Controls, Visualization, Log)
        right_pane = ttk.Frame(self.main_frame)
        right_pane.grid(row=0, column=1, sticky="nsew", rowspan=3, padx=5)
        self.main_frame.grid_columnconfigure(1, weight=1)

        control_frame = ttk.Frame(right_pane, padding="5")
        control_frame.pack(fill=tk.X, pady=2)
        self.setup_button = ttk.Button(
            control_frame, text="Setup Sim", command=self.setup_simulation
        )
        self.setup_button.pack(side="left", padx=2)
        self.run_step_button = ttk.Button(
            control_frame, text="Run Step", command=self.run_one_step, state=tk.DISABLED
        )
        self.run_step_button.pack(side="left", padx=2)
        self.run_all_button = ttk.Button(
            control_frame,
            text="Run All Steps",
            command=self.run_all_steps,
            state=tk.DISABLED,
        )
        self.run_all_button.pack(side="left", padx=2)
        self.compare_button = ttk.Button(
            control_frame,
            text="Show Comparison Plots",
            command=self.display_comparison_plots,
            state=tk.DISABLED,
        )
        self.compare_button.pack(side="left", padx=2)

        vis_pane = ttk.Labelframe(right_pane, text="Network Visualization", padding="5")
        vis_pane.pack(fill=tk.BOTH, expand=True, pady=2)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_pane)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        log_pane = ttk.Labelframe(right_pane, text="Simulation Log", padding="5")
        log_pane.pack(fill=tk.X, pady=2, ipady=2, side=tk.BOTTOM)
        self.log_text = scrolledtext.ScrolledText(
            log_pane, wrap=tk.WORD, height=8, width=80
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _toggle_ppp_params_visibility(self):
        placement_method = self.param_widgets["placement_method_var"].get()
        is_ppp = placement_method == "PPP"
        for widget_key in ["num_bss_label", "num_bss", "num_ues_label", "num_ues"]:
            if is_ppp:
                self.param_widgets[widget_key].grid_remove()
            else:
                self.param_widgets[widget_key].grid()
        for widget_key in ["lambda_bs_label", "lambda_bs", "lambda_ue_label", "lambda_ue"]:
            if not is_ppp:
                self.param_widgets[widget_key].grid_remove()
            else:
                self.param_widgets[widget_key].grid()

    def _populate_params_from_simparams(self):
        self.param_widgets["placement_method_var"].set(self.params.placement_method)
        self.param_widgets["rl_agent_type_var"].set(self.params.rl_agent_type)

        for name, widget in self.param_widgets.items():
            if name.endswith("_var") or name.endswith("_label"):
                continue
            if hasattr(self.params, name):
                widget.delete(0, tk.END)
                widget.insert(0, str(getattr(self.params, name)))

    def _update_simparams_from_gui(self):
        try:
            temp_params = SimParams()
            temp_params.placement_method = self.param_widgets["placement_method_var"].get()
            temp_params.rl_agent_type = self.param_widgets["rl_agent_type_var"].get()

            for name, widget in self.param_widgets.items():
                if name.endswith("_var") or name.endswith("_label"):
                    continue
                if not widget.winfo_ismapped():  # Skip hidden widgets
                    continue
                value_str = widget.get()
                if not value_str:
                    messagebox.showerror("Input Error", f"Parameter '{name}' cannot be empty.")
                    return False

                default_val_attr = getattr(SimParams(), name, None)
                if default_val_attr is None:
                    continue  # Should not happen if all fields correspond to SimParams

                if isinstance(default_val_attr, float):
                    value = float(value_str)
                elif isinstance(default_val_attr, int):
                    # Use math.ceil for robustness if float conversion happened
                    value = int(math.ceil(float(value_str)))
                else:
                    value = value_str
                setattr(temp_params, name, value)

            if temp_params.time_step_duration <= 0:
                messagebox.showerror("Input Error", "Time Step Duration must be greater than 0.")
                return False

            # Recalculate derived parameters
            temp_params.rb_bandwidth_hz = temp_params.rb_bandwidth_mhz * 1e6
            temp_params.ho_time_to_trigger_steps = (
                int(temp_params.ho_time_to_trigger_s / temp_params.time_step_duration)
                if temp_params.time_step_duration > 0
                else 0
            )

            self.params = temp_params
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric value entered: {e}")
            return False
        except Exception as e:
            messagebox.showerror(
                "Parameter Error",
                f"An unexpected error occurred while updating parameters: {e}",
            )
            return False

    def log_message(self, message):
        self.log_handler.log(message)  # Use the LogHandler

    def setup_simulation(self):
        if not self._update_simparams_from_gui():
            return

        # Set a fixed random seed for reproducibility for this run
        seed = 42  # Could be a GUI parameter too
        random.seed(seed)
        np.random.seed(seed)
        if TF_AVAILABLE_GLOBAL:
            import tensorflow as tf

            tf.random.set_seed(seed)

        self.log_message(
            f"Setting up simulation with agent: {self.params.rl_agent_type} (Seed: {seed})..."
        )
        try:
            self.simulation = Simulation(self.params, self.log_message)
            self.log_message("Simulation setup complete.")
            self.run_step_button.config(state=tk.NORMAL)
            self.run_all_button.config(state=tk.NORMAL)
            self.compare_button.config(state=tk.DISABLED)  # Disable until a run is complete
            self.update_visualization(initial_setup=True)
        except ImportError as e:
            messagebox.showerror(
                "RL Agent Error",
                f"Cannot initialize agent: {e}. Please ensure required libraries are installed.",
            )
            self.log_message(f"Error during setup: {e}")
            self.run_step_button.config(state=tk.DISABLED)
            self.run_all_button.config(state=tk.DISABLED)
        except Exception as e:
            self.log_message(f"Error during setup: {e}")
            messagebox.showerror("Setup Error", f"{e}")

    def run_one_step(self):
        if self.simulation:
            try:
                if self.simulation.run_step():
                    self.update_visualization()
                else:
                    self.log_message("Simulation finished (last step).")
                    self.run_step_button.config(state=tk.DISABLED)
                    self.run_all_button.config(state=tk.DISABLED)
                    self._finalize_run()
            except Exception as e:
                self.log_message(
                    f"Error during step {self.simulation.current_time_step}: {e}"
                )
                messagebox.showerror(
                    "Runtime Error",
                    f"Error during step {self.simulation.current_time_step}: {e}",
                )
                self.run_step_button.config(state=tk.DISABLED)
                self.run_all_button.config(state=tk.DISABLED)
        else:
            messagebox.showwarning("Not Setup", "Please setup simulation first.")

    def run_all_steps(self):
        if self.simulation:
            self.run_step_button.config(state=tk.DISABLED)
            self.run_all_button.config(state=tk.DISABLED)
            try:
                # Run remaining steps
                for _ in range(self.params.total_sim_steps - self.simulation.current_time_step):
                    if not self.simulation.run_step():
                        break  # Stop if simulation indicates it's done
                    self.update_visualization()
                    self.update_idletasks()  # Keep GUI responsive
                self.log_message("Simulation finished (all steps).")
                self._finalize_run()
            except Exception as e:
                self.log_message(f"Error during full run: {e}")
                messagebox.showerror("Runtime Error", f"Error during full run: {e}")
            finally:
                self.setup_button.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("Not Setup", "Please setup simulation first.")

    def _finalize_run(self):
        if self.simulation and self.simulation.metrics_history:
            current_agent_type = self.params.rl_agent_type
            self.all_runs_metrics[current_agent_type] = list(
                self.simulation.metrics_history
            )
            self.log_message(f"Metrics for {current_agent_type} stored.")
            self.display_final_metrics_summary(self.simulation.metrics_history)
            if len(self.all_runs_metrics) > 0:
                self.compare_button.config(state=tk.NORMAL)
        self.run_step_button.config(state=tk.DISABLED)
        self.run_all_button.config(state=tk.DISABLED)

    def update_visualization(self, initial_setup=False):
        if not self.simulation:
            if initial_setup:  # Clear and set initial plot area
                self.ax.clear()
                self.ax.set_xlim(0, self.params.sim_area_x)
                self.ax.set_ylim(0, self.params.sim_area_y)
                self.ax.set_xlabel("X (m)")
                self.ax.set_ylabel("Y (m)")
                self.ax.set_title("Network Setup")
                self.ax.grid(True)
                self.canvas.draw()
            return

        self.ax.clear()
        self.ax.set_xlim(0, self.params.sim_area_x)
        self.ax.set_ylim(0, self.params.sim_area_y)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        current_display_step = (
            self.simulation.current_time_step - 1
            if self.simulation.current_time_step > 0
            else 0
        )
        self.ax.set_title(
            f"Network - Step {current_display_step} - Agent: {self.params.rl_agent_type}"
        )
        self.ax.grid(True, linestyle=":", alpha=0.6)

        # Plot BSs
        bs_label_done = False
        for bs_id, bs_obj in self.simulation.bss.items():
            self.ax.plot(
                bs_obj.position[0],
                bs_obj.position[1],
                "ks",
                markersize=8,
                label="BS" if not bs_label_done else "",
            )
            self.ax.text(
                bs_obj.position[0] + 15,
                bs_obj.position[1] + 15,
                f"{bs_id}\nL:{bs_obj.load_factor_metric:.2f}",
                fontsize=7,
                color="k",
            )
            bs_label_done = True

        # Plot UEs and their connections
        ue_labels_done = {"satisfied": False, "not_satisfied": False, "no_conn": False}
        for ue_id, ue_obj in self.simulation.ues.items():
            ue_pos = ue_obj.position
            rb_count = len(ue_obj.rbs_from_bs1) + len(ue_obj.rbs_from_bs2)

            is_connected = ue_obj.serving_bs_1 or ue_obj.serving_bs_2
            if is_connected:
                if ue_obj.current_total_rate_mbps >= self.params.target_ue_throughput_mbps:
                    color, lk, mk = "g", "satisfied", "o"
                else:
                    color, lk, mk = "b", "not_satisfied", "^"
            else:
                color, lk, mk = "r", "no_conn", "x"

            label_str = f"UE ({lk.replace('_', ' ')})"
            self.ax.plot(
                ue_pos[0],
                ue_pos[1],
                marker=mk,
                color=color,
                markersize=6,
                label=label_str if not ue_labels_done[lk] else "",
            )
            self.ax.text(
                ue_pos[0],
                ue_pos[1] - 30,
                f"{ue_id[:3]}:{rb_count}",
                fontsize=6,
                color="dimgray",
            )
            if not ue_labels_done[lk]:
                ue_labels_done[lk] = True

            # Draw primary connection
            if ue_obj.serving_bs_1:
                bs1_pos = ue_obj.serving_bs_1.position
                self.ax.plot(
                    [ue_pos[0], bs1_pos[0]],
                    [ue_pos[1], bs1_pos[1]],
                    "k-",
                    alpha=0.7,
                    linewidth=1.5,
                )
            # Draw secondary connection (if different from primary)
            if ue_obj.serving_bs_2 and ue_obj.serving_bs_2 != ue_obj.serving_bs_1:
                bs2_pos = ue_obj.serving_bs_2.position
                self.ax.plot(
                    [ue_pos[0], bs2_pos[0]],
                    [ue_pos[1], bs2_pos[1]],
                    "c-",
                    alpha=0.5,
                    linewidth=1.0,
                )

            # Draw best non-serving candidate
            cand_count = 0
            for cand_bs_id, rsrp in ue_obj.best_bs_candidates:
                # Skip if it's currently serving (primary or secondary)
                if (ue_obj.serving_bs_1 and cand_bs_id == ue_obj.serving_bs_1.id) or (
                    ue_obj.serving_bs_2 and cand_bs_id == ue_obj.serving_bs_2.id
                ):
                    continue

                cand_bs_obj = self.simulation.bss.get(cand_bs_id)
                if cand_bs_obj:
                    cand_bs_pos = cand_bs_obj.position
                    self.ax.plot(
                        [ue_pos[0], cand_bs_pos[0]],
                        [ue_pos[1], cand_bs_pos[1]],
                        "m--",  # Magenta dashed for candidates
                        alpha=0.3,
                        linewidth=0.8,
                    )
                    cand_count += 1
                if cand_count >= 1:  # Only show the best non-serving candidate
                    break

        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            fontsize="xx-small",
            ncol=2,
        )
        self.canvas.draw()

    def display_final_metrics_summary(self, metrics_history_for_run):
        if not metrics_history_for_run:
            self.log_message("No metrics to display for current run.")
            return

        final_summary = metrics_history_for_run[-1]

        # Calculate averages over the entire run
        avg_ue_thr = (
            np.mean(
                [
                    m["avg_ue_throughput_mbps"]
                    for m in metrics_history_for_run
                    if m["num_connected_ues"] > 0
                ]
            )
            if any(m["num_connected_ues"] > 0 for m in metrics_history_for_run)
            else 0
        )
        avg_satisfied = (
            np.mean(
                [
                    m["percentage_satisfied_ues"]
                    for m in metrics_history_for_run
                    if m["num_connected_ues"] > 0
                ]
            )
            if any(m["num_connected_ues"] > 0 for m in metrics_history_for_run)
            else 0
        )
        avg_reward = np.mean([m["reward"] for m in metrics_history_for_run])
        avg_handovers_per_step = np.mean(
            [m["handovers_this_step"] for m in metrics_history_for_run]
        )

        summary_text = (
            f"--- Run Summary ({self.params.rl_agent_type}) ---\n"
            f"Total Steps: {final_summary['time_step']}\n"
            f"Cumulative HOs: {final_summary['total_handovers_cumulative']} (Avg per step: {avg_handovers_per_step:.2f})\n"
            f"Overall Avg UE Thr: {avg_ue_thr:.2f} Mbps\n"
            f"Overall Avg % Satisfied UEs: {avg_satisfied:.2f} %\n"
            f"Overall Avg Reward: {avg_reward:.3f}\n"
            f"Final Avg BS Load Factor: {final_summary['avg_bs_load_factor']:.2f}\n"
        )
        self.log_message(summary_text)

    def display_comparison_plots(self):
        if not self.all_runs_metrics:
            messagebox.showinfo(
                "No Comparison Data", "Run simulations with different agents first to compare."
            )
            return

        num_agents = len(self.all_runs_metrics)
        if num_agents == 0:
            return

        comp_window = tk.Toplevel(self)
        comp_window.title("RL Agent Comparison Plots")
        comp_window.geometry("1200x800")

        fig_comp, axs_comp = plt.subplots(2, 2, figsize=(11.5, 7.5))
        fig_comp.tight_layout(pad=4.0)

        metrics_to_plot = [
            ("avg_ue_throughput_mbps", "Avg UE Throughput (Mbps)"),
            ("percentage_satisfied_ues", "% Satisfied UEs"),
            ("avg_bs_load_factor", "Avg BS Load Factor"),
            ("reward", "Step Reward"),
        ]

        plot_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for i, (metric_key, plot_title) in enumerate(metrics_to_plot):
            ax = axs_comp[plot_coords[i]]
            for agent_name, history in self.all_runs_metrics.items():
                if not history:
                    continue
                time_steps = np.array([m["time_step"] for m in history])
                metric_values = np.array([m[metric_key] for m in history])
                ax.plot(
                    time_steps,
                    metric_values,
                    marker=".",
                    linestyle="-",
                    markersize=3,
                    label=agent_name,
                )

            ax.set_title(plot_title)
            ax.set_xlabel("Time Step")
            # Extract unit from title if present, otherwise use key
            ylabel_text = (
                plot_title.split("(")[-1].replace(")", "").strip()
                if "(" in plot_title
                else metric_key.replace("_", " ").title()
            )
            ax.set_ylabel(ylabel_text)
            ax.grid(True)
            ax.legend(fontsize="small")

        canvas_comp = FigureCanvasTkAgg(fig_comp, master=comp_window)
        canvas_comp.draw()
        canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)