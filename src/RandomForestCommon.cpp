#include "RandomForestCommon.h"

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
