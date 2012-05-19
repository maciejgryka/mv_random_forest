#include "RandomForestCommon.h"

#include <fstream>

using namespace RandomForest;

int RandomForest::CountCommas(const std::string& s) {
  int count = 0;
  for (std::string::const_iterator c = s.begin(); c != s.end(); ++c) {
    if (*c == ',') { ++count; }
  }
  return count;
}

RowVector RandomForest::ReadFloatVector(const std::string& line, int size) {
  RowVector rvec(size);
  std::stringstream ss(line);
  std::string value;
  for (int c = 0; c < size; c++) {
	  getline(ss, value, ',');
	  std::istringstream(value) >> rvec(c);
  }
  return rvec;
}

Matrix RandomForest::ReadFloatMatrix(const std::string& line, int height, int width) {
  Matrix mat(height, width);
  std::stringstream ss(line);
  std::string value;
  for (int r = 0; r < height; r++) {
	  for (int c = 0; c < width; c++) {
		  getline(ss, value, ',');
		  std::istringstream(value) >> mat(r, c);
	  }
  }
  return mat;
}

void RandomForest::SerializeMatrix(const Matrix& data, const std::string& fileName) {
  int rows(static_cast<int>(data.rows()));
  int cols(static_cast<int>(data.cols()));
  // open the file for writing
  std::ofstream out_file;
  out_file.open(fileName, std::ios::out);

  float temp;
	for (int rIt = 0; rIt < rows; rIt ++) {
		for (int cIt = 0; cIt < cols; cIt ++) {
      temp = data(rIt, cIt);
      out_file << temp;
      if (cIt < cols-1) { out_file << ","; }
		}
    out_file << "\n";
	}
	out_file.close();
}
