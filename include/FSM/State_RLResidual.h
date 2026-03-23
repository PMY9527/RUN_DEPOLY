// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/algorithms/algorithms.h"
#include "cmg_viz_shm.h"
#include <cmath>
#include <numeric>

class State_RLResidual : public FSMState
{
public:
    State_RLResidual(int state_mode, std::string state_string);

    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                // Policy step: robot update, obs compute (last_action from prev combined), policy act
                auto t0 = clock::now();
                env->step();
                auto t1 = clock::now();

                // CMG: AR/Non-AR forward
                auto& jp = env->robot->data.joint_pos;
                auto& jv = env->robot->data.joint_vel;
                auto cmd = isaaclab::observations_map()["keyboard_velocity_commands"](env.get(), {}); // velocity_commands if using joystick
                cmg->forward_ar( // forward_ar if using autoregressive cmg
                    {jp.data(), jp.data() + jp.size()},
                    {jv.data(), jv.data() + jv.size()},
                    {cmd.data(), cmd.data() + cmd.size()}
                );
                auto t2 = clock::now();
                auto qr = cmg->get_qref();

                // qref + raw_residual
                auto raw_residual = env->action_manager->action();
                std::vector<float> combined(raw_residual.size());
                for (size_t i = 0; i < combined.size(); ++i)
                    combined[i] = qr[i] + raw_residual[i];
                combined[25] = 0.0f;  // zero wrist pitch
                combined[26] = 0.0f;

                env->action_manager->process_action(combined);

                // // Print CMG qref and residual per joint every 0.5s
                // {
                //     static auto last_print = clock::now();
                //     auto now = clock::now();
                //     if (std::chrono::duration<double>(now - last_print).count() >= 0.5) {
                //         last_print = now;
                //         printf("\n--- CMG qref | Residual ---\n");
                //         for (size_t i = 0; i < qr.size(); ++i) {
                //             printf("  [%2zu] qref:%+.4f  res:%+.4f\n", i, qr[i], raw_residual[i]);
                //         }
                //     }
                // }

                // Print inference times every 1s
                // {
                //     static auto last_print = clock::now();
                //     auto now = clock::now();
                //     if (std::chrono::duration<double>(now - last_print).count() >= 1.0) {
                //         last_print = now;
                //         double policy_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                //         double cmg_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                //         printf("[RLResidual] policy: %.2f ms | cmg: %.2f ms\n", policy_ms, cmg_ms);
                //     }
                // }

                // Publish CMG data to shared memory for visualization
                if (cmg_viz.ok()) {
                    auto motion_ref = cmg->get_motion_ref();
                    std::vector<float> qref_vel(motion_ref.begin() + 29, motion_ref.end());
                    cmg_viz.write(
                        qr,                                                    // qref positions
                        qref_vel,                                              // qref velocities
                        {jp.data(), jp.data() + jp.size()},                    // actual positions
                        {jv.data(), jv.data() + jv.size()},                    // actual velocities
                        {cmd.data(), cmd.data() + cmd.size()},                 // velocity commands
                        raw_residual,                                          // policy residual
                        combined                                               // final combined
                    );
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<isaaclab::CMGRunner> cmg;
    CMGVizWriter cmg_viz;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLResidual)
