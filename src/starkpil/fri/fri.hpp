#ifndef FRI_HPP
#define FRI_HPP

#include "stark_info.hpp"
#include "proof_stark.hpp"
#include <cassert>
#include <vector>
#include "ntt_goldilocks.hpp"
#include "merklehash_goldilocks.hpp"
#include "merkleTreeGL.hpp"
#include "merkleTreeBN128.hpp"

template <typename ElementType>
class FRI
{
public:
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;

    static void fold(uint64_t step, Goldilocks::Element *pol, Goldilocks::Element *challenge, StarkInfo starkInfo);
    static void merkelize(uint64_t step, FRIProof<ElementType> &proof, Goldilocks::Element* pol, StarkInfo starkInfo, MerkleTreeType* treeFRI);
    static void proveQueries(uint64_t* friQueries, FRIProof<ElementType> &fproof, MerkleTreeType **trees, StarkInfo starkInfo);
    static void proveFRIQueries(uint64_t* friQueries, Goldilocks::Element* buffer, FRIProof<ElementType> &fproof, MerkleTreeType **treesFRI, StarkInfo starkInfo);
private:
    static vector<MerkleProof<ElementType>> queryPol(MerkleTreeType *trees[], uint64_t nTrees, uint64_t idx, ElementType* buff);
    static vector<MerkleProof<ElementType>> queryPol(MerkleTreeType *tree, uint64_t idx, ElementType* buff);
    static void polMulAxi(Goldilocks::Element *pol, uint64_t degree, Goldilocks::Element acc);
    static void evalPol(Goldilocks::Element* res, uint64_t res_idx, uint64_t degree, Goldilocks::Element* p, Goldilocks::Element *x);
    static void getTransposed(Goldilocks::Element *aux, Goldilocks::Element* pol, uint64_t degree, uint64_t trasposeBits);
};

template class FRI<RawFr::Element>;
template class FRI<Goldilocks::Element>;

#endif