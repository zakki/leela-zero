/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "GTP.h"
#include "GameState.h"
#include "NNCache.h"
#include "Network.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Utils.h"
#include "Zobrist.h"

#include "third_party/httplib.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "third_party/stb_image_resize.h"


using namespace Utils;

static void license_blurb() {
    printf(
        "Leela Zero %s  Copyright (C) 2017-2019  Gian-Carlo Pascutto and contributors\n"
        "This program comes with ABSOLUTELY NO WARRANTY.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see the COPYING file for details.\n\n",
        PROGRAM_VERSION);
}

static void calculate_thread_count_cpu(
    boost::program_options::variables_map& vm) {
    // If we are CPU-based, there is no point using more than the number of CPUs.
    auto cfg_max_threads = std::min(SMP::get_num_cpus(), size_t{MAX_CPUS});

    if (vm["threads"].as<unsigned int>() > 0) {
        auto num_threads = vm["threads"].as<unsigned int>();
        if (num_threads > cfg_max_threads) {
            myprintf("Clamping threads to maximum = %d\n", cfg_max_threads);
            num_threads = cfg_max_threads;
        }
        cfg_num_threads = num_threads;
    } else {
        cfg_num_threads = cfg_max_threads;
    }
}

#ifdef USE_OPENCL
static void calculate_thread_count_gpu(
    boost::program_options::variables_map& vm) {
    auto cfg_max_threads = size_t{MAX_CPUS};

    // Default thread count : GPU case
    // 1) if no args are given, use batch size of 5 and thread count of (batch size) * (number of gpus) * 2
    // 2) if number of threads are given, use batch size of (thread count) / (number of gpus) / 2
    // 3) if number of batches are given, use thread count of (batch size) * (number of gpus) * 2
    auto gpu_count = cfg_gpus.size();
    if (gpu_count == 0) {
        // size of zero if autodetect GPU : default to 1
        gpu_count = 1;
    }

    if (vm["threads"].as<unsigned int>() > 0) {
        auto num_threads = vm["threads"].as<unsigned int>();
        if (num_threads > cfg_max_threads) {
            myprintf("Clamping threads to maximum = %d\n", cfg_max_threads);
            num_threads = cfg_max_threads;
        }
        cfg_num_threads = num_threads;

        if (vm["batchsize"].as<unsigned int>() > 0) {
            cfg_batch_size = vm["batchsize"].as<unsigned int>();
        } else {
            cfg_batch_size =
                (cfg_num_threads + (gpu_count * 2) - 1) / (gpu_count * 2);

            // no idea why somebody wants to use threads less than the number of GPUs
            // but should at least prevent crashing
            if (cfg_batch_size == 0) {
                cfg_batch_size = 1;
            }
        }
    } else {
        if (vm["batchsize"].as<unsigned int>() > 0) {
            cfg_batch_size = vm["batchsize"].as<unsigned int>();
        } else {
            cfg_batch_size = 5;
        }

        cfg_num_threads =
            std::min(cfg_max_threads, cfg_batch_size * gpu_count * 2);
    }

    if (cfg_num_threads < cfg_batch_size) {
        printf(
            "Number of threads = %d must be no smaller than batch size = %d\n",
            cfg_num_threads, cfg_batch_size);
        exit(EXIT_FAILURE);
    }
}
#endif

static void parse_commandline(const int argc, const char* const argv[]) {
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description gen_desc("Generic options");
    gen_desc.add_options()
        ("help,h", "Show commandline options.")
        ("gtp,g", "Enable GTP mode.")
        ("threads,t", po::value<unsigned int>()->default_value(0),
                      "Number of threads to use. Select 0 to let leela-zero pick a reasonable default.")
        ("playouts,p", po::value<int>(),
                       "Weaken engine by limiting the number of playouts. "
                       "Requires --noponder.")
        ("visits,v", po::value<int>(),
                     "Weaken engine by limiting the number of visits.")
        ("lagbuffer,b", po::value<int>()->default_value(cfg_lagbuffer_cs),
                        "Safety margin for time usage in centiseconds.")
        ("resignpct,r", po::value<int>()->default_value(cfg_resignpct),
                        "Resign when winrate is less than x%.\n"
                        "-1 uses 10% but scales for handicap.")
        ("weights,w", po::value<std::string>()->default_value(cfg_weightsfile),
                      "File with network weights.")
        ("logfile,l", po::value<std::string>(),
                      "File to log input/output to.")
        ("quiet,q", "Disable all diagnostic output.")
        ("timemanage", po::value<std::string>()->default_value("auto"),
                       "[auto|on|off|fast|no_pruning] Enable time management features.\n"
                       "auto = no_pruning when using -n, otherwise on.\n"
                       "on = Cut off search when the best move can't change"
                       ", but use full time if moving faster doesn't save time.\n"
                       "fast = Same as on but always plays faster.\n"
                       "no_pruning = For self play training use.\n")
        ("noponder", "Disable thinking on opponent's time.")
        ("benchmark", "Test network and exit. Default args:\n-v3200 --noponder "
                      "-m0 -t1 -s1.")
#ifndef USE_CPU_ONLY
        ("cpu-only", "Use CPU-only implementation and do not use OpenCL device(s).")
#endif
        ;
#ifdef USE_OPENCL
    po::options_description gpu_desc("OpenCL device options");
    gpu_desc.add_options()
        ("gpu", po::value<std::vector<int>>(),
                "ID of the OpenCL device(s) to use (disables autodetection).")
        ("full-tuner", "Try harder to find an optimal OpenCL tuning.")
        ("tune-only", "Tune OpenCL only and then exit.")
        ("batchsize", po::value<unsigned int>()->default_value(0),
                      "Max batch size.  Select 0 to let leela-zero pick a reasonable default.")
#ifdef USE_HALF
        ("precision", po::value<std::string>(),
                      "Floating-point precision (single/half/auto).\n"
                      "Default is to auto which automatically determines which one to use.")
#endif
        ;
#endif
    po::options_description selfplay_desc("Self-play options");
    selfplay_desc.add_options()
        ("noise,n", "Enable policy network randomization.")
        ("seed,s", po::value<std::uint64_t>(),
                   "Random number generation seed.")
        ("dumbpass,d", "Don't use heuristics for smarter passing.")
        ("randomcnt,m", po::value<int>()->default_value(cfg_random_cnt),
                        "Play more randomly the first x moves.")
        ("randomvisits", po::value<int>()->default_value(cfg_random_min_visits),
                         "Don't play random moves if they have <= x visits.")
        ("randomtemp", po::value<float>()->default_value(cfg_random_temp),
                       "Temperature to use for random move selection.");
#ifdef USE_TUNER
    po::options_description tuner_desc("Tuning options");
    tuner_desc.add_options()
        ("puct", po::value<float>())
        ("logpuct", po::value<float>())
        ("logconst", po::value<float>())
        ("softmax_temp", po::value<float>())
        ("fpu_reduction", po::value<float>())
        ("ci_alpha", po::value<float>());
#endif
    // These won't be shown, we use them to catch incorrect usage of the
    // command line.
    po::options_description ignore("Ignored options");
#ifndef USE_OPENCL
    ignore.add_options()
        ("batchsize", po::value<unsigned int>()->default_value(1),
                      "Max batch size.");
#endif
    po::options_description h_desc("Hidden options");
    h_desc.add_options()
        ("arguments", po::value<std::vector<std::string>>());
    po::options_description visible;
    visible
        .add(gen_desc)
#ifdef USE_OPENCL
        .add(gpu_desc)
#endif
        .add(selfplay_desc)
#ifdef USE_TUNER
        .add(tuner_desc);
#else
        ;
#endif
    // Parse both the above, we will check if any of the latter are present.
    po::options_description all;
    all.add(visible).add(ignore).add(h_desc);
    po::positional_options_description p_desc;
    p_desc.add("arguments", -1);
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(all)
                      .positional(p_desc)
                      .run(),
                  vm);
        po::notify(vm);
    } catch (const boost::program_options::error& e) {
        printf("ERROR: %s\n", e.what());
        license_blurb();
        std::cout << visible << std::endl;
        exit(EXIT_FAILURE);
    }

    // Handle commandline options
    if (vm.count("help") || vm.count("arguments")) {
        auto ev = EXIT_SUCCESS;
        // The user specified an argument. We don't accept any, so explain
        // our usage.
        if (vm.count("arguments")) {
            for (auto& arg : vm["arguments"].as<std::vector<std::string>>()) {
                std::cout << "Unrecognized argument: " << arg << std::endl;
            }
            ev = EXIT_FAILURE;
        }
        license_blurb();
        std::cout << visible << std::endl;
        exit(ev);
    }

    if (vm.count("quiet")) {
        cfg_quiet = true;
    }

    if (vm.count("benchmark")) {
        cfg_quiet = true; // Set this early to avoid unnecessary output.
    }

#ifdef USE_TUNER
    if (vm.count("puct")) {
        cfg_puct = vm["puct"].as<float>();
    }
    if (vm.count("logpuct")) {
        cfg_logpuct = vm["logpuct"].as<float>();
    }
    if (vm.count("logconst")) {
        cfg_logconst = vm["logconst"].as<float>();
    }
    if (vm.count("softmax_temp")) {
        cfg_softmax_temp = vm["softmax_temp"].as<float>();
    }
    if (vm.count("fpu_reduction")) {
        cfg_fpu_reduction = vm["fpu_reduction"].as<float>();
    }
    if (vm.count("ci_alpha")) {
        cfg_ci_alpha = vm["ci_alpha"].as<float>();
    }
#endif

    if (vm.count("logfile")) {
        cfg_logfile = vm["logfile"].as<std::string>();
        myprintf("Logging to %s.\n", cfg_logfile.c_str());
        cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
    }

    cfg_weightsfile = vm["weights"].as<std::string>();
    if (vm["weights"].defaulted()
        && !boost::filesystem::exists(cfg_weightsfile)) {
        printf("A network weights file is required to use the program.\n");
        printf("By default, Leela Zero looks for it in %s.\n",
               cfg_weightsfile.c_str());
        exit(EXIT_FAILURE);
    }

    if (vm.count("gtp")) {
        cfg_gtp_mode = true;
    }

#ifdef USE_OPENCL
    if (vm.count("gpu")) {
        cfg_gpus = vm["gpu"].as<std::vector<int>>();
    }

    if (vm.count("full-tuner")) {
        cfg_sgemm_exhaustive = true;

        // --full-tuner auto-implies --tune-only.  The full tuner is so slow
        // that nobody will wait for it to finish befure running a game.
        // This simply prevents some edge cases from confusing other people.
        cfg_tune_only = true;
    }

    if (vm.count("tune-only")) {
        cfg_tune_only = true;
    }
#ifdef USE_HALF
    if (vm.count("precision")) {
        auto precision = vm["precision"].as<std::string>();
        if ("single" == precision) {
            cfg_precision = precision_t::SINGLE;
        } else if ("half" == precision) {
            cfg_precision = precision_t::HALF;
        } else if ("auto" == precision) {
            cfg_precision = precision_t::AUTO;
        } else {
            printf("Unexpected option for --precision, expecting single/half/auto\n");
            exit(EXIT_FAILURE);
        }
    }
    if (cfg_precision == precision_t::AUTO) {
        // Auto precision is not supported for full tuner cases.
        if (cfg_sgemm_exhaustive) {
            printf("Automatic precision not supported when doing exhaustive tuning\n");
            printf("Please add '--precision single' or '--precision half'\n");
            exit(EXIT_FAILURE);
        }
    }
#endif
    if (vm.count("cpu-only")) {
        cfg_cpu_only = true;
    }
#else
    cfg_cpu_only = true;
#endif

    if (cfg_cpu_only) {
        calculate_thread_count_cpu(vm);
    } else {
#ifdef USE_OPENCL
        calculate_thread_count_gpu(vm);
        myprintf("Using OpenCL batch size of %d\n", cfg_batch_size);
#endif
    }
    myprintf("Using %d thread(s).\n", cfg_num_threads);

    if (vm.count("seed")) {
        cfg_rng_seed = vm["seed"].as<std::uint64_t>();
        if (cfg_num_threads > 1) {
            myprintf("Seed specified but multiple threads enabled.\n");
            myprintf("Games will likely not be reproducible.\n");
        }
    }
    myprintf("RNG seed: %llu\n", cfg_rng_seed);

    if (vm.count("noponder")) {
        cfg_allow_pondering = false;
    }

    if (vm.count("noise")) {
        cfg_noise = true;
    }

    if (vm.count("dumbpass")) {
        cfg_dumbpass = true;
    }

    if (vm.count("playouts")) {
        cfg_max_playouts = vm["playouts"].as<int>();
        if (!vm.count("noponder")) {
            printf("Nonsensical options: Playouts are restricted but "
                   "thinking on the opponent's time is still allowed. "
                   "Add --noponder if you want a weakened engine.\n");
            exit(EXIT_FAILURE);
        }

        // 0 may be specified to mean "no limit"
        if (cfg_max_playouts == 0) {
            cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
        }
    }

    if (vm.count("visits")) {
        cfg_max_visits = vm["visits"].as<int>();

        // 0 may be specified to mean "no limit"
        if (cfg_max_visits == 0) {
            cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
        }
    }

    if (vm.count("resignpct")) {
        cfg_resignpct = vm["resignpct"].as<int>();
    }

    if (vm.count("randomcnt")) {
        cfg_random_cnt = vm["randomcnt"].as<int>();
    }

    if (vm.count("randomvisits")) {
        cfg_random_min_visits = vm["randomvisits"].as<int>();
    }

    if (vm.count("randomtemp")) {
        cfg_random_temp = vm["randomtemp"].as<float>();
    }

    if (vm.count("timemanage")) {
        auto tm = vm["timemanage"].as<std::string>();
        if (tm == "auto") {
            cfg_timemanage = TimeManagement::AUTO;
        } else if (tm == "on") {
            cfg_timemanage = TimeManagement::ON;
        } else if (tm == "off") {
            cfg_timemanage = TimeManagement::OFF;
        } else if (tm == "fast") {
            cfg_timemanage = TimeManagement::FAST;
        } else if (tm == "no_pruning") {
            cfg_timemanage = TimeManagement::NO_PRUNING;
        } else {
            printf("Invalid timemanage value.\n");
            exit(EXIT_FAILURE);
        }
    }
    if (cfg_timemanage == TimeManagement::AUTO) {
        cfg_timemanage =
            cfg_noise ? TimeManagement::NO_PRUNING : TimeManagement::ON;
    }

    if (vm.count("lagbuffer")) {
        int lagbuffer = vm["lagbuffer"].as<int>();
        if (lagbuffer != cfg_lagbuffer_cs) {
            myprintf("Using per-move time margin of %.2fs.\n",
                     lagbuffer / 100.0f);
            cfg_lagbuffer_cs = lagbuffer;
        }
    }
    if (vm.count("benchmark")) {
        // These must be set later to override default arguments.
        cfg_allow_pondering = false;
        cfg_benchmark = true;
        cfg_noise = false; // Not much of a benchmark if random was used.
        cfg_random_cnt = 0;
        cfg_rng_seed = 1;
        cfg_timemanage = TimeManagement::OFF; // Reliable number of playouts.

        if (!vm.count("playouts") && !vm.count("visits")) {
            cfg_max_visits = 3200; // Default to self-play and match values.
        }
    }

    // Do not lower the expected eval for root moves that are likely not
    // the best if we have introduced noise there exactly to explore more.
    cfg_fpu_root_reduction = cfg_noise ? 0.0f : cfg_fpu_reduction;

    auto out = std::stringstream{};
    for (auto i = 1; i < argc; i++) {
        out << " " << argv[i];
    }
    if (!vm.count("seed")) {
        out << " --seed " << cfg_rng_seed;
    }
    cfg_options_str = out.str();
}

static void initialize_network() {
    auto network = std::make_unique<Network>();
    auto playouts = std::min(cfg_max_playouts, cfg_max_visits);
    network->initialize(playouts, cfg_weightsfile);

    GTP::initialize(std::move(network));
}

// Setup global objects after command line has been parsed
void init_global_objects() {
    thread_pool.initialize(cfg_num_threads);

    // Use deterministic random numbers for hashing
    auto rng = std::make_unique<Random>(5489);
    Zobrist::init_zobrist(*rng);

    // Initialize the main thread RNG.
    // Doing this here avoids mixing in the thread_id, which
    // improves reproducibility across platforms.
    Random::get_Rng().seedrandom(cfg_rng_seed);

    Utils::create_z_table();

    initialize_network();
}

void benchmark(GameState& game) {
    game.set_timecontrol(0, 1, 0, 0); // Set infinite time.
    game.play_textmove("b", "r16");
    game.play_textmove("w", "d4");
    game.play_textmove("b", "c3");

    auto search = std::make_unique<UCTSearch>(game, *GTP::s_network);
    game.set_to_move(FastBoard::WHITE);
    search->think(FastBoard::WHITE);
}

#if 1
const int color_map[] = {
    0x5548C1, 0x564AC2, 0x584CC4, 0x594EC6, 0x5A4FC7, 0x5B51C9, 0x5C53CA, 0x5D55CC,
    0x5E57CD, 0x5F58CF, 0x605AD0, 0x615CD1, 0x625ED3, 0x635FD4, 0x6461D6, 0x6663D7,
    0x6765D8, 0x6866DA, 0x6968DB, 0x6A6ADC, 0x6B6BDD, 0x6C6DDF, 0x6D6FE0, 0x6E71E1,
    0x6F72E2, 0x7074E3, 0x7176E4, 0x7377E6, 0x7479E7, 0x757BE8, 0x767CE9, 0x777EEA,
    0x787FEB, 0x7981EC, 0x7A83ED, 0x7B84EE, 0x7D86EE, 0x7E87EF, 0x7F89F0, 0x808BF1,
    0x818CF2, 0x828EF3, 0x838FF3, 0x8491F4, 0x8592F5, 0x8794F6, 0x8895F6, 0x8997F7,
    0x8A98F8, 0x8B9AF8, 0x8C9BF9, 0x8D9DF9, 0x8E9EFA, 0x90A0FA, 0x91A1FB, 0x92A2FB,
    0x93A4FC, 0x94A5FC, 0x95A6FD, 0x96A8FD, 0x97A9FD, 0x99AAFE, 0x9AACFE, 0x9BADFE,
    0x9CAEFE, 0x9DB0FF, 0x9EB1FF, 0x9FB2FF, 0xA0B3FF, 0xA2B5FF, 0xA3B6FF, 0xA4B7FF,
    0xA5B8FF, 0xA6B9FF, 0xA7BAFF, 0xA8BBFF, 0xA9BDFF, 0xABBEFF, 0xACBFFF, 0xADC0FF,
    0xAEC1FF, 0xAFC2FF, 0xB0C3FF, 0xB1C4FE, 0xB2C5FE, 0xB3C6FE, 0xB4C7FD, 0xB6C8FD,
    0xB7C9FD, 0xB8C9FC, 0xB9CAFC, 0xBACBFC, 0xBBCCFB, 0xBCCDFB, 0xBDCEFA, 0xBECEFA,
    0xBFCFF9, 0xC0D0F9, 0xC1D1F8, 0xC2D1F7, 0xC3D2F7, 0xC4D3F6, 0xC5D3F5, 0xC6D4F5,
    0xC7D4F4, 0xC8D5F3, 0xC9D6F2, 0xCAD6F2, 0xCBD7F1, 0xCCD7F0, 0xCDD8EF, 0xCED8EE,
    0xCFD9ED, 0xD0D9EC, 0xD1D9EB, 0xD2DAEA, 0xD3DAE9, 0xD4DAE8, 0xD5DBE7, 0xD6DBE6,
    0xD6DBE5, 0xD7DCE4, 0xD8DCE3, 0xD9DCE2, 0xDADCE1, 0xDBDCE0, 0xDBDCDE, 0xDCDDDD,
    0xDDDCDC, 0xDEDCDB, 0xDFDCD9, 0xE0DBD8, 0xE1DBD6, 0xE2DAD5, 0xE3D9D3, 0xE3D9D2,
    0xE4D8D1, 0xE5D8CF, 0xE6D7CE, 0xE7D6CC, 0xE7D6CB, 0xE8D5C9, 0xE9D4C8, 0xE9D3C6,
    0xEAD3C5, 0xEBD2C3, 0xEBD1C2, 0xECD0C0, 0xECCFBE, 0xEDCFBD, 0xEDCEBB, 0xEECDBA,
    0xEECCB8, 0xEFCBB7, 0xEFCAB5, 0xF0C9B4, 0xF0C8B2, 0xF0C7B1, 0xF1C6AF, 0xF1C5AD,
    0xF1C4AC, 0xF2C3AA, 0xF2C2A9, 0xF2C1A7, 0xF2BFA6, 0xF3BEA4, 0xF3BDA2, 0xF3BCA1,
    0xF3BB9F, 0xF3B99E, 0xF3B89C, 0xF3B79A, 0xF3B699, 0xF3B497, 0xF3B396, 0xF3B294,
    0xF3B093, 0xF3AF91, 0xF3AE8F, 0xF3AC8E, 0xF3AB8C, 0xF3A98B, 0xF3A889, 0xF2A788,
    0xF2A586, 0xF2A485, 0xF2A283, 0xF1A181, 0xF19F80, 0xF19E7E, 0xF09C7D, 0xF09A7B,
    0xF0997A, 0xEF9778, 0xEF9677, 0xEE9475, 0xEE9274, 0xED9172, 0xED8F71, 0xEC8D6F,
    0xEC8C6E, 0xEB8A6C, 0xEB886B, 0xEA8669, 0xEA8568, 0xE98366, 0xE88165, 0xE87F63,
    0xE77E62, 0xE67C61, 0xE57A5F, 0xE5785E, 0xE4765C, 0xE3745B, 0xE27259, 0xE17058,
    0xE06F57, 0xE06D55, 0xDF6B54, 0xDE6953, 0xDD6751, 0xDC6550, 0xDB634E, 0xDA614D,
    0xD95F4C, 0xD85D4B, 0xD75B49, 0xD65848, 0xD55647, 0xD45445, 0xD25244, 0xD15043,
    0xD04E42, 0xCF4B40, 0xCE493F, 0xCD473E, 0xCB443D, 0xCA423B, 0xC9403A, 0xC83D39,
    0xC63B38, 0xC53837, 0xC43636, 0xC23334, 0xC13033, 0xC02D32, 0xBE2A31, 0xBD2730,
    0xBB242F, 0xBA212E, 0xB91D2D, 0xB7192C, 0xB6142B, 0xB40F29, 0xB30828, 0xB10127,
};
#else
const int color_map[] = {
    0x000000, 0x050004, 0x090008, 0x0D010D, 0x110111, 0x140114, 0x160117, 0x19011A,
    0x1B011E, 0x1C0221, 0x1E0225, 0x1F0228, 0x20022C, 0x210230, 0x220233, 0x220337,
    0x23033A, 0x24033E, 0x240341, 0x240344, 0x250347, 0x25044A, 0x26044D, 0x260450,
    0x270453, 0x270456, 0x280458, 0x28045B, 0x29045D, 0x2A0560, 0x2A0562, 0x2B0565,
    0x2C0567, 0x2C056A, 0x2C056D, 0x2C0571, 0x2B0676, 0x29067B, 0x250682, 0x1E0789,
    0x0F0793, 0x070E92, 0x07158D, 0x061A87, 0x061F82, 0x06227D, 0x062578, 0x052873,
    0x052B6E, 0x052D6A, 0x052F66, 0x053162, 0x04335E, 0x04355B, 0x043658, 0x043855,
    0x043952, 0x043B50, 0x043C4D, 0x043D4B, 0x043F49, 0x034047, 0x034146, 0x034244,
    0x034443, 0x034541, 0x034640, 0x03473E, 0x03483D, 0x04493B, 0x044A3A, 0x044C38,
    0x044D37, 0x044E35, 0x044F33, 0x045032, 0x045130, 0x04522E, 0x04542C, 0x04552A,
    0x045628, 0x045726, 0x045824, 0x045922, 0x045A20, 0x045B1E, 0x055D1C, 0x055E1A,
    0x055F18, 0x056016, 0x056114, 0x056212, 0x056310, 0x05640F, 0x05650E, 0x05660E,
    0x05680D, 0x05690D, 0x056A0C, 0x056B0B, 0x056C0A, 0x056D08, 0x056E06, 0x096F05,
    0x0F7005, 0x147105, 0x187205, 0x1D7306, 0x227306, 0x267406, 0x2B7506, 0x2F7606,
    0x337606, 0x387706, 0x3C7706, 0x407806, 0x457806, 0x497906, 0x4D7906, 0x527A06,
    0x567A06, 0x5A7A06, 0x5F7B06, 0x637B06, 0x677B06, 0x6B7B06, 0x6F7C06, 0x737C06,
    0x787C06, 0x7C7C06, 0x807C06, 0x857C06, 0x8A7B07, 0x8F7B07, 0x947A07, 0x997A07,
    0x9F7908, 0xA47808, 0xAA7608, 0xB07508, 0xB77309, 0xBD7109, 0xC46F09, 0xCA6C0A,
    0xD1690A, 0xD8660A, 0xDF620B, 0xE65E0B, 0xED590B, 0xF3540C, 0xF4561B, 0xF55824,
    0xF5592C, 0xF55B32, 0xF55D37, 0xF65F3C, 0xF66240, 0xF66443, 0xF66646, 0xF66849,
    0xF66A4C, 0xF76B51, 0xF76D55, 0xF76F5B, 0xF77060, 0xF87266, 0xF8736C, 0xF87573,
    0xF87679, 0xF9787F, 0xF97986, 0xF97B8C, 0xF97C92, 0xF97D98, 0xF97F9E, 0xF980A4,
    0xF982AA, 0xF983AF, 0xF984B5, 0xF985BA, 0xF987BF, 0xF988C4, 0xF989C9, 0xF98BCE,
    0xF98CD3, 0xFA8DD8, 0xFA8EDC, 0xFA90E0, 0xFA91E5, 0xFA92E9, 0xFA94EC, 0xFA95F0,
    0xFA96F4, 0xFA98F8, 0xF99AFA, 0xF69DFA, 0xF3A1FA, 0xF1A4FB, 0xEFA7FB, 0xEDAAFB,
    0xEBACFB, 0xE9AFFB, 0xE8B1FB, 0xE7B3FB, 0xE6B5FB, 0xE5B7FC, 0xE4B9FC, 0xE4BBFC,
    0xE3BDFC, 0xE3BFFC, 0xE3C0FC, 0xE3C2FC, 0xE3C4FC, 0xE3C5FC, 0xE3C7FC, 0xE4C8FC,
    0xE4CAFC, 0xE4CBFD, 0xE5CDFD, 0xE5CEFD, 0xE6D0FD, 0xE7D1FD, 0xE7D2FD, 0xE8D4FD,
    0xE9D5FD, 0xE9D6FD, 0xEAD8FD, 0xEAD9FD, 0xEADBFD, 0xEADCFD, 0xEBDEFD, 0xEBE0FD,
    0xEBE1FE, 0xEBE3FE, 0xEBE4FE, 0xEBE6FE, 0xEBE7FE, 0xEBE9FE, 0xEBEAFE, 0xECECFE,
    0xECEDFE, 0xECEFFE, 0xECF0FE, 0xEDF2FE, 0xEDF3FE, 0xEEF5FE, 0xEEF6FE, 0xEFF7FE,
    0xF0F9FE, 0xF1FAFE, 0xF2FBFE, 0xF4FCFE, 0xF6FDFF, 0xF8FEFF, 0xFBFFFF, 0xFFFFFF,
};
#endif

static char* create_image(const FastBoard::vertex_t* board, const float* data, int* len) {
#if 0
    float min = 0.0f;
    float max = 0.10f;
#else
    float min = 10e10;
    float max = -10e10;
    for (auto i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        auto v = data[i];
        if (v < min)
            min = v;
        if (v > max)
            max = v;
    }
#endif
    auto buf = std::unique_ptr<unsigned char[]>(new unsigned char[BOARD_SIZE * BOARD_SIZE * 3]);
    for (auto i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        auto v = (int)((data[i] - min) / (max - min) * 255.0f + 0.5f);
        auto n = std::min(std::max(v, 0), 255);
        buf[i * 3 + 0] = (color_map[n] >> 16) & 0xff;
        buf[i * 3 + 1] = (color_map[n] >> 8) & 0xff;
        buf[i * 3 + 2] = color_map[n] & 0xff;
    }
#if 0
    unsigned char *png = stbi_write_png_to_mem(buf.get(), 3 * BOARD_SIZE,
      BOARD_SIZE, BOARD_SIZE, 3, len);
#else
    const auto scale = 5;
    auto out = std::unique_ptr<unsigned char[]>(new unsigned char[BOARD_SIZE * BOARD_SIZE * 3 * scale * scale]);
    stbir_resize_uint8(buf.get(), BOARD_SIZE, BOARD_SIZE, 0,
      out.get(), BOARD_SIZE * scale, BOARD_SIZE * scale, 0, 3);
    const auto per_line = BOARD_SIZE * scale;
    for (auto y = 0; y < BOARD_SIZE; y++) {
        for (auto x = 0; x < BOARD_SIZE; x++) {
            auto idx = y * BOARD_SIZE + x;
            if (board[idx] == FastBoard::EMPTY) continue;
            unsigned char stone = board[idx] == FastBoard::BLACK ? 0x33 : 0xff;
            unsigned char stone2 = board[idx] == FastBoard::BLACK ? 0x0 : 0xcc;
            for (auto oy = 1; oy <= 3; oy++) {
                for (auto ox = 1; ox <= 3; ox++) {
                    out[((y * scale + oy) * per_line + x * scale + ox) * 3]     = stone;
                    out[((y * scale + oy) * per_line + x * scale + ox) * 3 + 1] = stone;
                    out[((y * scale + oy) * per_line + x * scale + ox) * 3 + 2] = stone;
                }
            }
            //out[((y * scale + 0) * per_line + x * scale + 2) * 3] = stone;
            //out[((y * scale + 0) * per_line + x * scale + 2) * 3 + 1] = stone;
            //out[((y * scale + 0) * per_line + x * scale + 2) * 3 + 2] = stone;
            //out[((y * scale + 2) * per_line + x * scale + 0) * 3] = stone;
            //out[((y * scale + 2) * per_line + x * scale + 0) * 3 + 1] = stone;
            //out[((y * scale + 2) * per_line + x * scale + 0) * 3 + 2] = stone;
            out[((y * scale + 2) * per_line + x * scale + 4) * 3] = stone2;
            out[((y * scale + 2) * per_line + x * scale + 4) * 3 + 1] = stone2;
            out[((y * scale + 2) * per_line + x * scale + 4) * 3 + 2] = stone2;
            out[((y * scale + 1) * per_line + x * scale + 4) * 3] = stone2;
            out[((y * scale + 1) * per_line + x * scale + 4) * 3 + 1] = stone2;
            out[((y * scale + 1) * per_line + x * scale + 4) * 3 + 2] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 2) * 3] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 2) * 3 + 1] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 2) * 3 + 2] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 3) * 3] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 3) * 3 + 1] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 3) * 3 + 2] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 4) * 3] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 4) * 3 + 1] = stone2;
            out[((y * scale + 0) * per_line + x * scale + 4) * 3 + 2] = stone2;
        }
    }
    unsigned char *png = stbi_write_png_to_mem(out.get(), 3 * BOARD_SIZE * scale,
      BOARD_SIZE * scale, BOARD_SIZE * scale,
      3, len);
#endif
    return (char*) png;
}

void start_http_server() {
    extern int get_history_no(FastBoard::vertex_t color);
    extern const FastBoard::vertex_t* get_board_data_history(int history_pos);
    extern const float* get_policy_data_history(int history_pos);
    extern const float* get_value_data_history(int history_pos);
    std::thread t([]{
            using namespace httplib;
            Server svr;
            svr.Get("/", [](const Request& req, Response& res) {
                res.set_content(R"(
<head>
<style>
body{
background: #cfc;
}
img {
transform: scaleY(-1);
}
</style>
</head>
<body>
<h1>Leela Zero</h1>
<h2>Policy #0</h2>
<img id="policy-b0" src="/policy/b/0" width="190" height="190">
<img id="policy-w0" src="/policy/w/0" width="190" height="190">
<h2>Policy #1</h2>
<img id="policy-b1" src="/policy/b/1" width="190" height="190">
<img id="policy-w1" src="/policy/w/1" width="190" height="190">
<h2>Value</h2>
<img id="value-b" src="/value/b" width="190" height="190">
<img id="value-w" src="/value/w" width="190" height="190">
</div>
<script>
  setInterval(()=>{
    let t = new Date().getTime();
    document.getElementById("policy-b0").src = "/policy/b/0?_t=" + t;
    document.getElementById("policy-b1").src = "/policy/b/1?_t=" + t;
    document.getElementById("policy-w0").src = "/policy/w/0?_t=" + t;
    document.getElementById("policy-w1").src = "/policy/w/1?_t=" + t;
    document.getElementById("value-b").src = "/value/b?_t=" + t;
    document.getElementById("value-w").src = "/value/w?_t=" + t;
  }, 100);
</script>
</body>
                )", "text/html");
            });
            svr.Get(R"(/policy/(b|w)/(\d+))", [](const Request& req, Response& res) {
                auto color = req.matches[1].str() == "b" ? FastBoard::BLACK : FastBoard::WHITE;
                auto history_no = get_history_no(color);
                auto num = std::atoi(req.matches[2].str().c_str());
                auto board = get_board_data_history(history_no);
                auto data = get_policy_data_history(history_no) + num * BOARD_SIZE * BOARD_SIZE;
                int len;
                auto png = create_image(board, data, &len);
                if (png == NULL) return;
                res.set_content((char*)png, len, "image/png");
                STBIW_FREE(png);
            });
            svr.Get(R"(/value/(b|w))", [](const Request& req, Response& res) {
                auto color = req.matches[1].str() == "b" ? FastBoard::BLACK : FastBoard::WHITE;
                auto history_no = get_history_no(color);
                auto num = std::atoi(req.matches[2].str().c_str());
                auto board = get_board_data_history(history_no);
                auto data = get_value_data_history(history_no);
                int len;
                auto png = create_image(board, data, &len);
                if (png == NULL) return;
                res.set_content((char*)png, len, "image/png");
                STBIW_FREE(png);
            });
            svr.listen("localhost", 1919);
    });
    t.detach();
}

int main(int argc, char* argv[]) {
    // Set up engine parameters
    GTP::setup_default_parameters();
    parse_commandline(argc, argv);

    // Disable IO buffering as much as possible
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    std::cin.setf(std::ios::unitbuf);

    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
#ifndef _WIN32
    setbuf(stdin, nullptr);
#endif

    if (!cfg_gtp_mode && !cfg_benchmark) {
        license_blurb();
    }

    init_global_objects();

    auto maingame = std::make_unique<GameState>();

    /* set board limits */
    maingame->init_game(BOARD_SIZE, KOMI);

    if (cfg_benchmark) {
        cfg_quiet = false;
        benchmark(*maingame);
        return 0;
    }

    if (true) {
        start_http_server();
    }
   
    for (;;) {
        if (!cfg_gtp_mode) {
            maingame->display_state();
            std::cout << "Leela: ";
        }

        auto input = std::string{};
        if (std::getline(std::cin, input)) {
            Utils::log_input(input);
            GTP::execute(*maingame, input);
        } else {
            // eof or other error
            std::cout << std::endl;
            break;
        }

        // Force a flush of the logfile
        if (cfg_logfile_handle) {
            fclose(cfg_logfile_handle);
            cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
        }
    }

    return 0;
}
