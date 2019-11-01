#include <ros/ros.h>

#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/Handler.h"
#include "semantic_slam/OdometryHandler.h"
#include "semantic_slam/PosePresenter.h"
#include "semantic_slam/TrajectoryPresenter.h"

#include <mutex>
#include <condition_variable>
#include <csignal>
#include <iostream>

namespace {
    std::function<void(int)> shutdown_handler;
    void signal_handler(int signal) { shutdown_handler(signal); }
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "semslam");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    boost::shared_ptr<FactorGraph> graph(new FactorGraph);

    // Handlers notify us that they modified the graph via a condition variable notify
    // TODO this seems messy
    std::mutex mutex;
    boost::shared_ptr<std::condition_variable> cv = boost::make_shared<std::condition_variable>();

    std::vector<boost::shared_ptr<Handler>> handlers;

    handlers.emplace_back(new OdometryHandler(graph, cv));

    for (auto& h : handlers) {
        h->setup();
    }

    std::vector<boost::shared_ptr<Presenter>> presenters;

    presenters.emplace_back(new PosePresenter(graph, cv));
    presenters.emplace_back(new TrajectoryPresenter(graph, cv));

    for (auto& p : presenters) {
        p->setup();
    }

    // Start message handling threads
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // install a signal handler so we can stop the cv waiting on CTRL+C
    shutdown_handler = [&](int signal) { 
        std::cout << " Shutting down " << std::endl;
        ros::shutdown();
        cv->notify_all(); 
        // why doesn't this work to shut it down damnit
        // have to do this --
        exit(1);
    };

    std::signal(SIGINT, signal_handler);


    while (ros::ok()) {
        // Process the graph when one of our handlers modifies it
        std::unique_lock<std::mutex> lock(mutex);

        // cv->wait(lock, [&]() { return graph->modified(); });
        cv->wait(lock);

        for (auto& h : handlers) {
            h->update();
        }

        // ROS_INFO_STREAM("Notified!");

        bool solve_succeeded = graph->solve();

        if (solve_succeeded) {
            ROS_INFO_STREAM("Solve succeeded");
            for (auto& p : presenters) p->present();
        } else {
            ROS_INFO_STREAM("Solve failed");
            return 1;
        }
    }
}