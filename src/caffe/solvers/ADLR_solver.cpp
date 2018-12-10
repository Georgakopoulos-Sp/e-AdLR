#include <vector>

#include "caffe/sgd_solvers.hpp"
#include <fstream>

namespace caffe {

	template <typename Dtype>
		void ADLRSolver<Dtype>::ADLRPreSolve() {
			/* 
			// Add the extra history entries for ADLR after those from
			// SGDSolver::PreSolve
			const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
			for (int i = 0; i < net_params.size(); ++i) {
			const vector<int>& shape = net_params[i]->shape();
			this->history_.push_back(
			shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
			}
			*/
		}

#ifndef CPU_ONLY
	template <typename Dtype>
		void ADLR_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

	template <typename Dtype>
		void ADLRSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
			const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
			const vector<float>& net_params_lr = this->net_->params_lr();
			Dtype momentum = this->param_.momentum();
			//			Dtype local_rate = rate * net_params_lr[param_id];
			Dtype local_rate = rate;
			switch (Caffe::mode()) {
				case Caffe::CPU: {
					caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
								net_params[param_id]->cpu_diff(), momentum,
								this->history_[param_id]->mutable_cpu_data());
					caffe_copy(net_params[param_id]->count(),
								this->history_[param_id]->cpu_data(),
								net_params[param_id]->mutable_cpu_diff());
					break;
				}
				case Caffe::GPU: {
#ifndef CPU_ONLY
				// START
					if (this->iter_<4){
						local_rate=rate;
						this->my_adaptive_learning_rate=local_rate;
						this->my_learning_rate[param_id]=local_rate;
						this->my_adaptive_learning_rate_previous=this->my_adaptive_learning_rate;

						caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
									net_params[param_id]->gpu_diff(), momentum,
									this->history_[param_id]->mutable_gpu_data());
										 // copy
						caffe_copy(net_params[param_id]->count(),
									this->history_[param_id]->gpu_data(),
									net_params[param_id]->mutable_gpu_diff());
							if (this->iter_==3){
								caffe_copy(net_params[param_id]->count(),
										net_params[param_id]->gpu_diff(),
										this->old2_[param_id]->mutable_gpu_diff());

							}
							if (this->iter_==2){
								caffe_copy(net_params[param_id]->count(),
										 net_params[param_id]->gpu_diff(),
										 this->old1_[param_id]->mutable_gpu_diff());
							}
						}
						else
						{

							Dtype mytmp2_now=0;
							Dtype mytmp12=0;
							Dtype g1;
							Dtype g2;

							Dtype gg1=14;
							Dtype gg2=15;

							caffe_copy(net_params[param_id]->count(),
									 net_params[param_id]->gpu_diff(),
									 this->orig_history_[param_id]->mutable_gpu_diff());


							 int old1_ind;
							 int old2_ind;

							 int old1_min_ind;
							 int old2_min_ind;
							 int old_min_this_ind;

											 // ------ ---- A try with Thrust library for min max element ---------------------//
											 Dtype *new_min_old1=&gg1;
											 Dtype *new_max_old1=&gg2;
											 //		std::cout<<"G1 "<<gg1<<std::endl;
											 my_caffe_gpu_new_min(net_params[param_id]->count(), this->old1_[param_id]->gpu_diff(), new_max_old1, new_min_old1, this->old1_[param_id]->mutable_gpu_diff());
											 //		std::cout<<"Thrust Min "<<*new_min_old1<<std::endl;
											 //		std::cout<<"Thrust Max "<<*new_max_old1<<std::endl;
											 my_caffe_gpu_reg(net_params[param_id]->count(), this->old1_[param_id]->gpu_diff(), *new_max_old1, *new_min_old1, this->old1_[param_id]->mutable_gpu_diff());

											 my_caffe_gpu_new_min(net_params[param_id]->count(), this->old2_[param_id]->gpu_diff(), new_max_old1, new_min_old1, this->old2_[param_id]->mutable_gpu_diff());
											 my_caffe_gpu_reg(net_params[param_id]->count(), this->old2_[param_id]->gpu_diff(), *new_max_old1, *new_min_old1, this->old2_[param_id]->mutable_gpu_diff());

											 my_caffe_gpu_new_min(net_params[param_id]->count(), this->orig_history_[param_id]->gpu_diff(), new_max_old1, new_min_old1, this->orig_history_[param_id]->mutable_gpu_diff());
											 my_caffe_gpu_reg(net_params[param_id]->count(), this->orig_history_[param_id]->gpu_diff(), *new_max_old1, *new_min_old1, this->orig_history_[param_id]->mutable_gpu_diff());

											 //							End of Thrust try										//




								caffe_gpu_dot(net_params[param_id]->count(), 
										 this->old2_[param_id]->gpu_diff(), 
										 this->orig_history_[param_id]->gpu_diff(), &mytmp2_now);
								caffe_gpu_dot(net_params[param_id]->count(), 
										 this->old1_[param_id]->gpu_diff(), 
										 this->old2_[param_id]->gpu_diff(), &mytmp12);

								Dtype sumold2, sumold1, sumhist;
								caffe_gpu_dot(this->orig_history_[param_id]->count(), this->orig_history_[param_id]->gpu_diff(), this->orig_history_[param_id]->mutable_gpu_diff(), &sumhist);
								caffe_gpu_dot(this->orig_history_[param_id]->count(), this->old1_[param_id]->gpu_diff(), this->old1_[param_id]->mutable_gpu_diff(), &sumold1);
								caffe_gpu_dot(this->orig_history_[param_id]->count(), this->old2_[param_id]->gpu_diff(), this->old2_[param_id]->mutable_gpu_diff(), &sumold2);
								
								mytmp2_now = mytmp2_now / (sumhist*sumold2);
								mytmp12 = mytmp12 / (sumold1*sumold2);
//								mytmp2_now = mytmp2_now / (2*this->orig_history_[param_id]->count());
//									mytmp12 = mytmp12 / (2*this->orig_history_[param_id]->count());

								g1 = mytmp12;
								g2 = mytmp2_now;
								this->my_learning_rate[param_id] = g2*0.001  + g1*0.0001 + this->my_learning_rate[param_id];
								Dtype thelearning = this->my_adaptive_learning_rate;

								caffe_copy(net_params[param_id]->count(),
										this->old2_[param_id]->gpu_diff(),
										this->old1_[param_id]->mutable_gpu_diff());

/*
								caffe_gpu_axpby(net_params[param_id]->count(), 
										this->my_adaptive_learning_rate,
											//thelearning,
											//				local_rate,
											net_params[param_id]->gpu_diff(), momentum,
											this->history_[param_id]->mutable_gpu_data());
											 // copy
								caffe_copy(net_params[param_id]->count(),
										this->history_[param_id]->gpu_data(),
										net_params[param_id]->mutable_gpu_diff());
*/
								
								ADLR_update_gpu(net_params[param_id]->count(),
						        net_params[param_id]->mutable_gpu_diff(),
						        this->history_[param_id]->mutable_gpu_data(),
						        momentum, this->my_adaptive_learning_rate);
											 
								caffe_copy(net_params[param_id]->count(),
										 this->history_[param_id]->gpu_data(),
										 net_params[param_id]->mutable_gpu_diff());
   							    caffe_copy(net_params[param_id]->count(),
										 net_params[param_id]->gpu_diff(),
										 this->old2_[param_id]->mutable_gpu_diff());



											 //			std::cout<<this->net_->params().size()<<std::endl;
											 //			std::cout<<param_id<<std::endl;
								if ((this->net_->params().size()-1)==param_id){
									this->ofs_Lp.open ("learning_param_ADLR.txt", std::ofstream::out | std::fstream::app);
												 //			this->ofs_Lp<<mytmp2_now;
									this->ofs_Lp<<thelearning;
									this->ofs_Lp<<"\n";
									this->ofs_Lp.close();
								}
								if ((this->net_->params().size()-1)==param_id){
									Dtype tmp=0;
									for (int jj=2;jj<4;jj++){
										tmp=tmp+this->my_learning_rate[jj];
									}
									this->my_adaptive_learning_rate=tmp/2;
									if (this->my_adaptive_learning_rate<=0){
										std::cout<<"Do reset"<<std::endl;
										this->my_adaptive_learning_rate=rate*0.01;
										for (int jj=0;jj<2;jj++){
											this->my_learning_rate[jj]=rate*0.01;
										}
									}
								}

							}
#else
							NO_GPU;
#endif
							break;
						}
						default:
						LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
					}
			}

INSTANTIATE_CLASS(ADLRSolver);
REGISTER_SOLVER_CLASS(ADLR);

}  // namespace caffe
