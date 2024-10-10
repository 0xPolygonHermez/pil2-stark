#ifndef MERKLETREEGL
#define MERKLETREEGL

#include "goldilocks_base_field.hpp"
#include "poseidon_goldilocks.hpp"
#include "zklog.hpp"
#include <math.h>

class MerkleTreeGL
{
private:
    Goldilocks::Element getElement(uint64_t idx, uint64_t subIdx);
    void genMerkleProof(Goldilocks::Element *proof, uint64_t idx, uint64_t offset, uint64_t n);
    void calculateRootFromProof(Goldilocks::Element (&value)[4], std::vector<std::vector<Goldilocks::Element>> mp, uint64_t idx, uint64_t offset);

public:
    MerkleTreeGL(){};
    MerkleTreeGL(uint64_t _arity, bool custom, Goldilocks::Element *tree);
    MerkleTreeGL(uint64_t _arity, bool custom, uint64_t _height, uint64_t _width, Goldilocks::Element *_source, bool allocate = true);
    ~MerkleTreeGL();

    uint64_t numNodes;
    uint64_t height;
    uint64_t width;

    Goldilocks::Element *nodes;
    Goldilocks::Element *source;

    bool isSourceAllocated = false;
    bool isNodesAllocated = false;

    uint64_t arity;
    bool custom;

    uint64_t nFieldElements = HASH_SIZE;

    uint64_t getNumSiblings(); 
    uint64_t getMerkleTreeWidth(); 
    uint64_t getMerkleProofSize(); 
    uint64_t getMerkleProofLength();

    uint64_t getNumNodes(uint64_t height);
    void getRoot(Goldilocks::Element *root);
    void copySource(Goldilocks::Element *_source);
    void setSource(Goldilocks::Element *_source);

    void getGroupProof(Goldilocks::Element *proof, uint64_t idx);
    
    bool verifyGroupProof(Goldilocks::Element* root, std::vector<std::vector<Goldilocks::Element>> mp, uint64_t idx, std::vector<std::vector<Goldilocks::Element>> v);

    void merkelize();
};

#endif