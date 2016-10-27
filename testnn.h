
#include <random>
#include <memry>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

namespace mynn{
	template <typename T, std::size_t N>
	class variable{
		using namespace thrust;
	public:
		template <class U>
		// constant range
		variable():this->host(N),this->device(N){
			fill(host.begin(), host.end(), 0);
		}
		variable(U rand):this->host(N),this->device(N){
			thrust::generate(this->host.begin(), this->host.end(), rand);
		}

		// use befor forward
		void set_device(){
			thrust::copy(this->host.begin(), this->host.end(), this->device.begin());
		}

		//use after backward
		void set_host(){
			thrust::copy(this->device.begin(), this->device.end(), this->host.begin());
		}

		const device_vector<T>& forward() const {
			return device;
		}

	private:
		host_vector<T> host;
		device_vector<T> device;
	};

	template <typename T,std::size_t N,class U>
	inline std::shared_ptr<variable<T,N> > make_variable(U rand = random::uniform_real_distribution()){
		return std::make_shared<variable<T,N>>(rand);
	}
	
	template <typename T, std::size_t N, std::size_t M>
	class linear{
		using namespace thrust;
	public:
		template <class U>
		linear(U rand = random::uniform_real_distribution()):this->host(N*M),this->device(N*M){
			thrust::generate(this->host.begin(), this->host.end(), rand);
			outputer
			this->set_device();

		}

		const variable<T,N>& operator()(const variable<T,N>& input){////////////////////////////////////////
			inputer = *input;
			return outputer;
		}

		// use befor forward
		void set_device(){
			thrust::copy(this->host.begin(), this->host.end(), this->device.begin());
		}

		//use after backward
		void set_host(){
			thrust::copy(this->device.begin(), this->device.end(), this->host.begin());
		}
		
		const device_vector<T>& forward() {
			inputer->forward
			return outputer.forward();
		}

	private:
		host_vector<T> host;
		device_vector<T> device;
		const variable<T,N> * inputer;
		variable<T,M> outputer;
	};
}
