//
// Created by depaulsmiller on 9/9/20.
//
#include <chrono>
#include <algorithm>
#include <vector>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <set>
#include <execution>

bool comp(const std::pair<unsigned, unsigned> &a, const std::pair<unsigned, unsigned> &b) {
    return a.first < b.first;
}

struct CompClass {
    CompClass(){}
    ~CompClass(){}
    bool operator()(const std::pair<unsigned, unsigned> &a, const std::pair<unsigned, unsigned> &b){
        return comp(a,b);
    }

};

int main() {

    int sizeOfTest = 512 * 136;

    std::vector<std::pair<unsigned, unsigned>> vec;
    vec.reserve(sizeOfTest);
    for (int i = 0; i < sizeOfTest; i++) {
        vec.push_back({rand(), 1});
    }
    std::vector<std::pair<unsigned, unsigned >> vec2;
    vec2.reserve(sizeOfTest);
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par_unseq , vec.begin(), vec.end(), comp);
    auto iter = vec.begin();
    while (iter != vec.end()) {
        vec2.push_back(*iter);
        auto iter2 = iter + 1;
        while (iter2 != vec.end() && iter2->first == iter->first) {
            ++iter2;
        }
        iter = iter2;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;

    std::cerr << "Sorting" << std::endl;
    std::cerr << dur.count() * 1e3 << " ms" << std::endl;
    std::cerr << vec.size() / 1e6 / dur.count() << std::endl;
    std::cerr << vec2.size() << " from " << vec.size() << std::endl;

    vec.clear();
    for (int i = 0; i < sizeOfTest; i++) {
        vec.push_back({rand(), 1});
    }

    vec2.clear();

    std::unordered_map<unsigned, std::pair<unsigned, unsigned>> map;

    start = std::chrono::high_resolution_clock::now();

    for(auto &e: vec){
        map[e.first] = e;
    }
    for(auto &e : map){
        vec2.push_back(e.second);
    }

    end = std::chrono::high_resolution_clock::now();
    dur = end - start;

    std::cerr << "Map" << std::endl;

    std::cerr << dur.count() * 1e3 << " ms" << std::endl;

    std::cerr << vec2.size() << " from " << vec.size() << std::endl;

    vec.clear();
    for (int i = 0; i < sizeOfTest; i++) {
        vec.push_back({rand(), 1});
    }

    vec2.clear();
    /*std::set<std::pair<unsigned, unsigned>, comp> s;
    start = std::chrono::high_resolution_clock::now();
    for(auto& e: vec){
        s.insert(e);
    }
    for(auto& e: s){
        vec2.push_back(e);
    }
    end = std::chrono::high_resolution_clock::now();
    dur = end - start;

    std::cerr << "Tree Set" << std::endl;

    std::cerr << dur.count() * 1e3 << " ms" << std::endl;

    std::cerr << vec2.size() << " from " << vec.size() << std::endl;
*/
    return 0;

}