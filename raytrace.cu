#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "canvas.h"

#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

class figure
{
public:
	int R, G, B;
	int type; // 1 = sphere | 2 = box | 3 = tetra
	double x1, y1, z1;
	double x2, y2, z2;
	double x3, y3, z3;
	double x4, y4, z4;
	double Rad; // For sphere
	figure(int type, int R, int G, int B,
		double x1, double y1, double z1,
		double x2, double y2, double z2,
		double x3, double y3, double z3,
		double x4, double y4, double z4, double Rad)
	{
		this->R = R;
		this->G = G;
		this->B = B;
		this->type = type;
		this->x1 = x1; this->y1 = y1; this->z1 = z1;
		this->x2 = x2; this->y2 = y2; this->z2 = z2;
		this->x3 = x3; this->y3 = y3; this->z3 = z3;
		this->x4 = x4; this->y4 = y4; this->z4 = z4;
		this->Rad = Rad;
	}
	figure() {}
};

__host__ figure* scene_objects(std::string filename, size_t* figure_count)
{
	size_t fig_count = 0;
	std::ifstream obj_file;
	obj_file.open(filename);
	if (!obj_file.is_open())
	{
		std::cout << "Error occured while tried to open " << filename << std::endl;
		exit(-1);
	}
	std::string tmp_object;
	while (getline(obj_file, tmp_object)) fig_count++;
	if (!fig_count)
	{
		std::cout << "Error: no objects encountered." << std::endl;
		exit(-1);
	}
	obj_file.clear();
	obj_file.seekg(0);
	figure* res = new figure[fig_count];
	for (size_t i = 0; i < fig_count; ++i)
	{
		obj_file >> tmp_object;
		if (tmp_object == "sphere")
		{
			int R, G, B;
			double x, y, z, Rad;
			obj_file >> R; obj_file >> G; obj_file >> B;
			obj_file >> x; obj_file >> y; obj_file >> z;
			obj_file >> Rad;
			res[i] = figure(1, R, G, B, x, y, z,
				0, 0, 0, 0, 0, 0, 0, 0, 0, Rad);
		}
		else
		{
			std::cout << "Error: unknown figure encountered." << std::endl;
			exit(-1);
		}
	}
	obj_file.close();
	*figure_count = fig_count;
	return res;
}

__host__ void scene_props(std::string filename,
	double* light,
	double* camera,
	double* upvector,
	double* screen_normal,
	double& screen_distance,
	double& view_depth,
	int& screen_width,
	int& screen_height)
{
	std::ifstream props_file;
	props_file.open(filename);
	if (!props_file.is_open())
	{
		std::cout << "Error occured while tried to open " << filename << std::endl;
		exit(-1);
	}
	std::string title;
	for (int i = 0; i < 8; ++i)
	{
		props_file >> title;
		switch (title.size())
		{
		case 6: //camera
			props_file >> camera[0];
			props_file >> camera[1];
			props_file >> camera[2];
			break;
		case 5: //light
			props_file >> light[0];
			props_file >> light[1];
			props_file >> light[2];
			break;
		case 7: //normal_
			props_file >> screen_normal[0];
			props_file >> screen_normal[1];
			props_file >> screen_normal[2];
			break;
		case 8: //upvector
			props_file >> upvector[0];
			props_file >> upvector[1];
			props_file >> upvector[2];
			break;
		case 11: //screen_dist
			props_file >> screen_distance;
			break;
		case 10: //view_depth
			props_file >> view_depth;
			break;
		case 12: //screen_width
			props_file >> screen_width;
			break;
		case 13: //screen_height
			props_file >> screen_height;
			break;
		default:
			std::cout << "Error: unknown title encountered." << std::endl;
			exit(-1);
		}
	}
	props_file.close();
}

__host__ void to_length(double* vec, double length)
{
	double K = sqrt(pow(length, 2) / (pow(vec[0], 2) + pow(vec[1], 2) + pow(vec[2], 2)));
	vec[0] *= K; vec[1] *= K; vec[2] *= K;
}

__host__ std::pair<int, int> optimal_dimension(int screen_width, int screen_height, int max_threads)
{
	int choosen_w = 105;
	if (screen_width % 5 != 0)
	{
		std::cout << "Error: unsupported image resolution." << std::endl;
		std::cout << "Please add manually needed block dimension to function 'optimal dimension'" << std::endl;
		exit(-1);
	}
	for (int dim = 100; dim >= 1; dim -= 5)
	{
		if (dim < choosen_w && screen_width % dim == 0 && screen_height * (screen_width / dim) < max_threads)
		{
			choosen_w = dim;
		}
	}
	return std::pair<int, int>(choosen_w, screen_height);
}

__host__ double* pack_objects(figure* objects, size_t fig_count)
{
	double* packed_objs = new double[fig_count * 17];
	for (size_t i = 0; i < fig_count; ++i)
	{
		packed_objs[i * 17] = objects[i].type;
		packed_objs[i * 17 + 1] = objects[i].R;
		packed_objs[i * 17 + 2] = objects[i].G;
		packed_objs[i * 17 + 3] = objects[i].B;
		packed_objs[i * 17 + 4] = objects[i].x1;
		packed_objs[i * 17 + 5] = objects[i].y1;
		packed_objs[i * 17 + 6] = objects[i].z1;
		if (objects[i].type == 1)
		{
			packed_objs[i * 17 + 16] = objects[i].Rad;
		}
		else
		{
			std::cout << "Error: unknown type" << std::endl;
			exit(-1);
		}
	}
	return packed_objs;
}

typedef struct
{
public:
	double log_value;
}GPU_log;

__global__ void raytrace_kernel(uint8_t* frame,
	double* objects,
	double* geometry_data, int thread_scope, GPU_log* log)
	//                                  geometry_data = light upvect ortsup camera lu_corner
{
	int x_scr_0 = blockIdx.x * thread_scope;
	int y_scr_0 = threadIdx.x;

	// Unpacking to shared memory block 
	__shared__ size_t f_count; f_count = geometry_data[17];
	__shared__ int scr_width; scr_width = geometry_data[15];
	__shared__ int scr_height; scr_height = geometry_data[16];
	__shared__ double camera[3]; camera[0] = geometry_data[9]; camera[1] = geometry_data[10]; camera[2] = geometry_data[11];
	__shared__ double light[3]; light[0] = geometry_data[0]; light[1] = geometry_data[1]; light[2] = geometry_data[2];
	__shared__ double lu_cor[3]; lu_cor[0] = geometry_data[12]; lu_cor[1] = geometry_data[13]; lu_cor[2] = geometry_data[14];
	__shared__ double ortsup[3]; ortsup[0] = geometry_data[6]; ortsup[1] = geometry_data[7]; ortsup[2] = geometry_data[8];
	__shared__ double upvect[3]; upvect[0] = geometry_data[3]; upvect[1] = geometry_data[4]; upvect[2] = geometry_data[5];

	for (int i = 0; i < thread_scope + 1; ++i)
	{
		int x_scr = x_scr_0 + i;
		int y_scr = y_scr_0;
		double x_phys = lu_cor[0] + x_scr * ortsup[0] + y_scr * upvect[0];
		double y_phys = lu_cor[1] + x_scr * ortsup[1] + y_scr * upvect[1];
		double z_phys = lu_cor[2] + x_scr * ortsup[2] + y_scr * upvect[2];
		double trace_ray[3] = { x_phys - camera[0], y_phys - camera[1], z_phys - camera[2] };
		int R, G, B;
		double surface_normal[3] = { 0,0,0 };
		double intersection[3] = { 0,0,0 };
		bool intersected = false;

		for (size_t i = 0; i < f_count; ++i)
		{
			int i_ = i * 17;
			if (objects[i_] == 1.0) // sphere
			{
				double A_ = trace_ray[0] * trace_ray[0] + trace_ray[1] * trace_ray[1] + trace_ray[2] * trace_ray[2];
				double B_ = 2 * (trace_ray[0] * camera[0] + trace_ray[1] * camera[1] + trace_ray[2] * camera[2] -
					trace_ray[0] * objects[i_ + 4] - trace_ray[1] * objects[i_ + 5] - trace_ray[2] * objects[i_ + 6]);
				double C_ = (camera[0] - objects[i_ + 4]) * (camera[0] - objects[i_ + 4]) +
					(camera[1] - objects[i_ + 5]) * (camera[1] - objects[i_ + 5]) +
					(camera[2] - objects[i_ + 6]) * (camera[2] - objects[i_ + 6]) - objects[i_ + 16] * objects[i_ + 16];
				double discr = B_ * B_ - 4 * A_ * C_;
				if (discr <= 0) continue;
				intersected = true;
				double param_1 = (-B_ - sqrt(discr)) / (2 * A_), param_2 = (-B_ + sqrt(discr)) / (2 * A_);
				(param_1 > param_2) ? param_1 = param_2 : param_2 = param_1;
				double tmp_intersection[3] = { camera[0] + trace_ray[0] * param_1, camera[1] + trace_ray[1] * param_1,
				camera[2] + trace_ray[2] * param_1 };
				if (i == 0 || ((pow(tmp_intersection[0] - camera[0], 2) + pow(tmp_intersection[1] - camera[1], 2) +
					pow(tmp_intersection[2] - camera[2], 2)) < (pow(intersection[0] - camera[0], 2 +
						pow(intersection[1] - camera[1], 2) + pow(intersection[2] - camera[2], 2)))))
				{
					for (int i = 0; i < 3; ++i) intersection[i] = tmp_intersection[i];
					for (int i = 0; i < 3; ++i) surface_normal[i] = intersection[i] - objects[i_ + 4 + i];
					R = objects[i_ + 1]; G = objects[i_ + 2]; B = objects[i_ + 3];
				}
			}
			else
			{
				/* Unknown shape type */
			}
		}
		if (intersected)
		{
			double light_vect[3] = { light[0] - intersection[0], light[1] - intersection[1], light[2] - intersection[2] };
			double cos_alpha = (light_vect[0] * surface_normal[0] + light_vect[1] * surface_normal[1] +
				light_vect[2] * surface_normal[2]) / (sqrt(light_vect[0] * light_vect[0] + light_vect[1] * light_vect[1] +
					light_vect[2] * light_vect[2]) * sqrt(surface_normal[0] * surface_normal[0] + surface_normal[1] * surface_normal[1] +
						surface_normal[2] * surface_normal[2]));
			if (cos_alpha + 0.2 < 0) cos_alpha = -0.2;
			R = static_cast<int>(R * pow(cos_alpha + 0.2, 1.5));
			G = static_cast<int>(G * pow(cos_alpha + 0.2, 1.5));
			B = static_cast<int>(B * pow(cos_alpha + 0.2, 1.5));
			frame[(scr_height - y_scr - 1) * scr_width * 3 + 3 * x_scr] = B;
			frame[(scr_height - y_scr - 1) * scr_width * 3 + 3 * x_scr + 1] = G;
			frame[(scr_height - y_scr - 1) * scr_width * 3 + 3 * x_scr + 2] = R;
		}
	}
}

__host__ int main(int argc, char* argv[])
{
	std::string objects = "objects.txt";
	std::string props = "properties.txt";
	std::string save_name = "image_1.bmp";

	/* ========================= */

	size_t fig_count;
	double* light = new double[3];
	double* camera = new double[3];
	double* screen_normal = new double[3];
	double* upvector = new double[3];
	double* ort_sup = new double[3];
	double* lu_corner = new double[3];
	double screen_dist, view_depth;
	int screen_width, screen_height;

	// Getting scene props & objects
	figure* Scene_objects = scene_objects(objects, &fig_count);
	scene_props(props, light, camera, upvector, screen_normal, screen_dist, view_depth, screen_width, screen_height);

	// Initializing image
	BMP_Image frame(screen_width, screen_height);
	for (int i = 0; i < frame.pixlen; ++i) frame.pixels[i] = 0;

	// Some geometry
	ort_sup[0] = upvector[1] * screen_normal[2] - upvector[2] * screen_normal[1];
	ort_sup[1] = upvector[2] * screen_normal[0] - upvector[0] * screen_normal[2];
	ort_sup[2] = upvector[0] * screen_normal[1] - upvector[1] * screen_normal[0];
	to_length(screen_normal, screen_dist);
	to_length(upvector, static_cast<double>(screen_height) / 2);
	to_length(ort_sup, static_cast<double>(screen_width) / 2);
	lu_corner[0] = camera[0] + screen_normal[0] + upvector[0] + ort_sup[0];
	lu_corner[1] = camera[1] + screen_normal[1] + upvector[1] + ort_sup[1];
	lu_corner[2] = camera[2] + screen_normal[2] + upvector[2] + ort_sup[2];
	to_length(upvector, 1.0);
	to_length(ort_sup, 1.0);
	for (int i = 0; i < 3; ++i) upvector[i] *= -1, ort_sup[i] *= -1;

	// Constructing thread-blocks grid
	if (cudaSetDevice(0) != cudaSuccess)
	{
		std::cout << "Error occured while tried to initialize GPU" << std::endl;
		exit(-1);
	}
	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, 0);
	int SM_s = device_properties.multiProcessorCount;
	int supported_threads = device_properties.maxThreadsPerMultiProcessor;
	int max_threads = SM_s * supported_threads;

	std::pair<int, int> blocks_dimension = optimal_dimension(screen_width, screen_height, max_threads);

	// Packing & transfering data to gpu
	double* device_geom_data;
	double* device_objs_data;
	double* packed_vect_data = new double[18]; // light upvect ortsup camera lu_corner
	double* packed_objs_data = pack_objects(Scene_objects, fig_count);
	uint8_t* device_canvas;
	GPU_log* device_log, * host_log = new GPU_log;
	packed_vect_data[0] = light[0]; packed_vect_data[1] = light[1]; packed_vect_data[2] = light[2];
	packed_vect_data[3] = upvector[0]; packed_vect_data[4] = upvector[1]; packed_vect_data[5] = upvector[2];
	packed_vect_data[6] = ort_sup[0]; packed_vect_data[7] = ort_sup[1]; packed_vect_data[8] = ort_sup[2];
	packed_vect_data[9] = camera[0]; packed_vect_data[10] = camera[1]; packed_vect_data[11] = camera[2];
	packed_vect_data[12] = lu_corner[0]; packed_vect_data[13] = lu_corner[1]; packed_vect_data[14] = lu_corner[2];
	packed_vect_data[15] = screen_width; packed_vect_data[16] = screen_height; packed_vect_data[17] = fig_count;

	if (cudaMalloc(&device_geom_data, 18 * sizeof(double)) != cudaSuccess ||
		cudaMalloc(&device_objs_data, (17 * fig_count) * sizeof(double)) != cudaSuccess ||
		cudaMalloc(&device_canvas, sizeof(uint8_t) * frame.pixlen) != cudaSuccess ||
		cudaMalloc(&device_log, sizeof(GPU_log)) != cudaSuccess)
	{
		std::cout << "Error occured while tried to allocate memory on GPU" << std::endl;
		exit(-1);
	}

	if (cudaMemcpy(device_geom_data, packed_vect_data, 18 * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(device_objs_data, packed_objs_data, (17 * fig_count) * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		std::cout << "Error occured while tried to transfer data to GPU" << std::endl;
		cudaFree(device_geom_data);
		cudaFree(device_objs_data);
		cudaFree(device_canvas);
		cudaFree(device_log);
		exit(-1);
	}

	// Invoking kernel
	raytrace_kernel << <screen_width / blocks_dimension.first, screen_height >> > (device_canvas,
		device_objs_data,
		device_geom_data,
		blocks_dimension.first,
		device_log);
	if (cudaMemcpy(frame.pixels, device_canvas, sizeof(uint8_t) * frame.pixlen, cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(host_log, device_log, sizeof(GPU_log), cudaMemcpyDeviceToHost))
	{
		std::cout << "Error occured while tried to transfer data from GPU" << std::endl;
		exit(-1);
	}

	///* Revise logvalue if needed */
	//std::cout << "Log value is: " << host_log->log_value << std::endl;
	///* ========================= */

	frame.save(save_name);

	if (cudaFree(device_geom_data) != cudaSuccess ||
		cudaFree(device_objs_data) != cudaSuccess ||
		cudaFree(device_canvas) != cudaSuccess ||
		cudaFree(device_log) != cudaSuccess)
	{
		std::cout << "Error occured while tried to free memory on GPU" << std::endl;
		exit(-1);
	}

	delete[] packed_objs_data;
	delete[] packed_vect_data;
	delete[] light;
	delete[] camera;
	delete[] upvector;
	delete[] ort_sup;
	delete[] screen_normal;
	delete[] lu_corner;
	delete host_log;

	std::cout << "Done." << std::endl;
	return 0;
}

