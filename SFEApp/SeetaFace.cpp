#include "SeetaFace.h"


SeetaFace::SeetaFace() {
	this->detector = new Detector("model/seeta_fd_frontal_v1.0.bin");
	this->point_detector = new seeta::FaceAlignment("model/seeta_fa_v1.1.bin");
	this->face_recognizer = new seeta::FaceIdentification("model/seeta_fr_v1.0.bin");
}

float* SeetaFace::NewFeatureBuffer() {
	return new float[this->face_recognizer->feature_size()];
}

bool SeetaFace::GetFeature(string filename, float* feat) {
	//如果有多张脸，输出第一张脸,把特征放入缓冲区feat，返回true
	//如果没有脸，输出false
	//read pic greyscale
	cv::Mat src_img = cv::imread(filename, 0);
	seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
	src_img_data.data = src_img.data;

	//read pic color
	cv::Mat src_img_color = cv::imread(filename, 1);
	seeta::ImageData src_img_data_color(src_img_color.cols, src_img_color.rows, src_img_color.channels());
	src_img_data_color.data = src_img_color.data;

	std::vector<seeta::FaceInfo> faces = this->detector->Detect(src_img_data);
	int32_t face_num = static_cast<int32_t>(faces.size());
	if (face_num == 0)
	{
		return false;
	}
	seeta::FacialLandmark points[5];
	this->point_detector->PointDetectLandmarks(src_img_data, faces[0], points);

	this->face_recognizer->ExtractFeatureWithCrop(src_img_data_color, points, feat);

	return true;
};

int SeetaFace::GetFeatureDims() {
	return this->face_recognizer->feature_size();
}

float SeetaFace::FeatureCompare(float* feat1, float* feat2) {
	return this->face_recognizer->CalcSimilarity(feat1, feat2);
}
