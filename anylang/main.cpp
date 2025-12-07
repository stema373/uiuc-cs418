#include "uselibpng.h"
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct Color {
  uint8_t r, g, b, a;
};

string trim(const string &s) {
  size_t start = 0;
  while (start < s.size() && isspace(static_cast<unsigned char>(s[start]))) {
    start++;
  }

  size_t end = s.size();
  while (end > start && isspace(static_cast<unsigned char>(s[end - 1]))) {
    end--;
  }

  return s.substr(start, end - start);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <filename>\n";
    return 1;
  }

  string filename = argv[1];

  ifstream file(filename);
  if (!file) {
    cerr << "Could not open file: " << filename << "\n";
    return 1;
  }

  string line;
  Image *img = nullptr;
  const char *fname;
  vector<pair<uint32_t, uint32_t>> pos;
  vector<Color> colors;
  while (getline(file, line)) {
    line = trim(line);
    if (line.empty())
      continue;

    istringstream iss(line);
    string keyword;
    iss >> keyword;

    if (keyword == "png") {
      string filename;
      uint32_t width, height;
      if (iss >> width >> height >> filename) {
        img = new Image(width, height);
        fname = filename.c_str();
      }
    } else if (keyword == "position") {
      int dummy;
      iss >> dummy;

      if (dummy != 2)
        continue;

      uint32_t x, y;

      pos.clear();

      while (iss >> x >> y) {
        pos.push_back({x, y});
      }
    } else if (keyword == "color") {
      int dummy;
      iss >> dummy;

      if (dummy != 4)
        continue;

      uint32_t r, g, b, a;

      colors.clear();

      while (iss >> r >> g >> b >> a) {
        colors.push_back({static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                          static_cast<uint8_t>(b), static_cast<uint8_t>(a)});
      }
    } else if (keyword == "drawPixels") {
      int n;
      iss >> n;

      for (int i = 0; i < n; i++) {
        pixel_t &p = (*img)[pos[i].second][pos[i].first];
        p.r = colors[i].r;
        p.g = colors[i].g;
        p.b = colors[i].b;
        p.a = colors[i].a;
      }
    }
  }

  if (img && fname) {
    img->save(fname);
    delete img;
  }

  return 0;
}
