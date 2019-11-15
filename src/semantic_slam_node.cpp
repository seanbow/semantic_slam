#include <ros/ros.h>

#include "semantic_slam/FactorGraph.h"
#include "semantic_slam/Handler.h"
#include "semantic_slam/SemanticMapper.h"
#include "semantic_slam/OdometryHandler.h"
#include "semantic_slam/PosePresenter.h"
#include "semantic_slam/TrajectoryPresenter.h"
#include "semantic_slam/ObjectMeshPresenter.h"

#include <rosfmt/rosfmt.h>
#include <fmt/ostream.h>

#include <mutex>
#include <condition_variable>
#include <csignal>
#include <iostream>
#include <chrono>

// namespace {
//     std::function<void(int)> shutdown_handler;
//     void signal_handler(int signal) { shutdown_handler(signal); }
// }

// class FactorGraphSupervisor {
// public:
//     FactorGraphSupervisor();

//     void addHandler(boost::shared_ptr<Handler> handler);
//     void addPresenter(boost::shared_ptr<Presenter> presenter);

//     void emplacePresenter(Presenter* presenter);
    
//     void start();
//     void stop();

// private:
//     ros::NodeHandle nh_;
//     ros::NodeHandle pnh_;

//     ros::AsyncSpinner msg_spinner_;

//     boost::shared_ptr<FactorGraph> graph_;

//     // Handlers notify us that they modified the graph via a condition variable notify
//     // TODO this seems messy
//     boost::shared_ptr<std::mutex> graph_mutex_;
//     boost::shared_ptr<std::condition_variable> graph_cv_;

//     std::vector<boost::shared_ptr<Handler>> handlers_;
//     std::vector<boost::shared_ptr<Presenter>> presenters_;
    
//     bool running_;
// };

// FactorGraphSupervisor::FactorGraphSupervisor()
//     : pnh_("~"),
//       msg_spinner_(1),
//       graph_(new FactorGraph),
//       running_(false)
// {
//     graph_mutex_ = boost::make_shared<std::mutex>();
//     graph_cv_ = boost::make_shared<std::condition_variable>();
// }

// void
// FactorGraphSupervisor::addHandler(boost::shared_ptr<Handler> handler) {
//     handlers_.push_back(handler);
//     handler->setGraph(graph_);
//     // handler->setCv(graph_cv_);
// }

// void
// FactorGraphSupervisor::addPresenter(boost::shared_ptr<Presenter> presenter) {
//     presenters_.push_back(presenter);
//     presenter->setGraph(graph_);
// }

// void FactorGraphSupervisor::emplacePresenter(Presenter* presenter) {
//     addPresenter(boost::shared_ptr<Presenter>(presenter));
// }

// void FactorGraphSupervisor::stop()
// {
//     msg_spinner_.stop();
//     running_ = false;
//     graph_cv_->notify_all();
// }

// void FactorGraphSupervisor::start()
// {
//     for (auto& h : handlers_) {
//         h->setup();
//     }

//     for (auto& p : presenters_) {
//         p->setup();
//     }

//     // Start message handling threads
//     msg_spinner_.start();

//     running_ = true;

//     while (ros::ok() && running_) {
//         // Process the graph when one of our handlers modifies it
//         // std::unique_lock<std::mutex> lock(*graph_mutex_);

//         // cv->wait(lock, [&]() { return graph->modified(); });
//         // graph_cv_->wait(lock);

//         for (auto& h : handlers_) {
//             h->update();
//         }

//         if (graph_->modified()) {
//             auto t1 = std::chrono::high_resolution_clock::now();
//             bool solve_succeeded = graph_->solve(true);
//             auto t2 = std::chrono::high_resolution_clock::now();
//             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

//             if (solve_succeeded) {
//                 ROS_INFO_STREAM(fmt::format("Solved {} nodes and {} edges in {:.2f} ms.",
//                                             graph_->num_nodes(), graph_->num_factors(), duration.count()/1000.0));
//                 for (auto& p : presenters_) p->present();
//             } else {
//                 ROS_INFO_STREAM("Graph solve failed");
//                 // return 1;
//             }

//             // auto tablenode = graph_->getNode<SE3Node>(Symbol('o', 1));
//             // if (tablenode) {
//             //     std::cout << "table pos = " << tablenode->pose().translation().transpose() << std::endl;
//             //     std::cout << "pointer = " << tablenode->pose().translation_data() << std::endl;
//             // }
//         }
//     }

//     msg_spinner_.stop();
// }

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "semslam");
    
    // FactorGraphSupervisor supervisor;

    auto odom_handler = boost::shared_ptr<OdometryHandler>(new OdometryHandler);

    auto mapper = boost::shared_ptr<SemanticMapper>(new SemanticMapper);
    mapper->setOdometryHandler(odom_handler);

    // Start message handling thread
    ros::AsyncSpinner message_spinner(1);
    message_spinner.start();
    
    // Setup and add presenters
    auto pose_presenter = util::allocate_aligned<PosePresenter>();
    auto trajectory_presenter = util::allocate_aligned<TrajectoryPresenter>();
    auto object_presenter = util::allocate_aligned<ObjectMeshPresenter>();

    mapper->addPresenter(pose_presenter);
    mapper->addPresenter(trajectory_presenter);
    mapper->addPresenter(object_presenter);

    mapper->start();

}