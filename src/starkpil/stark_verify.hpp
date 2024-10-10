#include "expressions_ctx.hpp"
#include "stark_info.hpp"
#include "merkleTreeGL.hpp"

bool starkVerify(FRIProof<Goldilocks::Element> &fproof, StarkInfo& starkInfo, ExpressionsBin& expressionsBin, Goldilocks::Element *verkey, Goldilocks::Element *publics, Goldilocks::Element* challenges_) {

    uint64_t friQueries[starkInfo.starkStruct.nQueries];
    
    Goldilocks::Element evals[starkInfo.evMap.size()  * FIELD_EXTENSION];
    for(uint64_t i = 0; i < starkInfo.evMap.size(); ++i) {
        memcpy(&evals[i * FIELD_EXTENSION], fproof.proof.evals[i].data(), FIELD_EXTENSION * sizeof(Goldilocks::Element));
    }

    Goldilocks::Element challenges[(starkInfo.challengesMap.size() + starkInfo.starkStruct.steps.size() + 1) * FIELD_EXTENSION];
    if(challenges_ == nullptr) {
        uint64_t c = 0;
        TranscriptGL transcript(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
        transcript.put(&verkey[0], 4);

        if(starkInfo.nPublics > 0) {
            if(!starkInfo.starkStruct.hashCommits) {
                transcript.put(&publics[0], starkInfo.nPublics);
            } else {
                Goldilocks::Element hash[4];
                TranscriptGL transcriptHash(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
                transcriptHash.put(&publics[0], starkInfo.nPublics);
                transcriptHash.getState(hash);
                transcript.put(hash, 4);
            }
        }

        for(uint64_t s = 1; s <= starkInfo.nStages + 1; ++s) {
            uint64_t nChallenges = std::count_if(starkInfo.challengesMap.begin(), starkInfo.challengesMap.end(),[s](const PolMap& c) { return c.stage == s; });
            for(uint64_t i = 0; i < nChallenges; ++i) {
                transcript.getField((uint64_t *)&challenges[(c++)*FIELD_EXTENSION]);
            }
            transcript.put(&fproof.proof.roots[s - 1][0], 4);
        }

        // Evals challenge
        transcript.getField((uint64_t *)&challenges[(c++)*FIELD_EXTENSION]);

        if(!starkInfo.starkStruct.hashCommits) {
            transcript.put(&evals[0], starkInfo.evMap.size()  * FIELD_EXTENSION);
        } else {
            Goldilocks::Element hash[4];
            TranscriptGL transcriptHash(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
            transcriptHash.put(&evals[0], starkInfo.evMap.size()  * FIELD_EXTENSION);
            transcriptHash.getState(hash);
            transcript.put(hash, 4);
        }

        // FRI challenges
        transcript.getField((uint64_t *)&challenges[(c++)*FIELD_EXTENSION]);
        transcript.getField((uint64_t *)&challenges[(c++)*FIELD_EXTENSION]);


        for (uint64_t step=0; step<starkInfo.starkStruct.steps.size(); step++) {
            transcript.getField((uint64_t *)&challenges[(c++)*FIELD_EXTENSION]);
            if (step < starkInfo.starkStruct.steps.size() - 1) {
                transcript.put(fproof.proof.fri.treesFRI[step].root.data(), 4);
            } else {
                uint64_t finalPolSize = (1<< starkInfo.starkStruct.steps[step].nBits);
                Goldilocks::Element finalPol[finalPolSize * FIELD_EXTENSION];
                for(uint64_t i = 0; i < finalPolSize; ++i) {
                    memcpy(&finalPol[i * FIELD_EXTENSION], fproof.proof.fri.pol[i].data(), FIELD_EXTENSION * sizeof(Goldilocks::Element));
                }

                if(!starkInfo.starkStruct.hashCommits) {
                    transcript.put(&finalPol[0],finalPolSize*FIELD_EXTENSION);
                } else {
                    Goldilocks::Element hash[4];
                    TranscriptGL transcriptHash(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
                    transcriptHash.put(&finalPol[0], finalPolSize*FIELD_EXTENSION);
                    transcriptHash.getState(hash);
                    transcript.put(hash, 4);
                }
            }
        }
        transcript.getField((uint64_t *)&challenges[(c++)*FIELD_EXTENSION]);

        zklog.trace(Goldilocks::toString(challenges[0]) + ", " + Goldilocks::toString(challenges[1]) + ", " + Goldilocks::toString(challenges[2]));

    } else {
        std::memcpy(challenges, challenges_, ((starkInfo.challengesMap.size() + starkInfo.starkStruct.steps.size() + 1) * FIELD_EXTENSION) * sizeof(Goldilocks::Element));
    }

    Goldilocks::Element *challenge = &challenges[(starkInfo.challengesMap.size() + starkInfo.starkStruct.steps.size()) * FIELD_EXTENSION];

    TranscriptGL transcriptPermutation(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
    transcriptPermutation.put(challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.steps[0].nBits);

    for(uint64_t i = 0; i < starkInfo.nStages + 1; ++i) {
        std::string section = "cm" + to_string(i + 1);
        uint64_t nCols = starkInfo.mapSectionsN[section];
        MerkleTreeGL *tree = new MerkleTreeGL(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, 1 << starkInfo.starkStruct.nBitsExt, nCols, NULL, false);

        for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
            bool res = tree->verifyGroupProof(&fproof.proof.roots[i][0], fproof.proof.fri.trees.polQueries[q][i].mp, friQueries[q], fproof.proof.fri.trees.polQueries[q][i].v);
            if(!res) {
                return false;
            }
        }
    }

    MerkleTreeGL *treeC = new MerkleTreeGL(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.nConstants, NULL, false);
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        bool res = treeC->verifyGroupProof(verkey, fproof.proof.fri.trees.polQueries[q][starkInfo.nStages + 1].mp, friQueries[q], fproof.proof.fri.trees.polQueries[q][starkInfo.nStages + 1].v);
        if(!res) {
            return false;
        }
    }

    // TODO: Verify Quotient Polynomial

    for (uint64_t step=1; step< starkInfo.starkStruct.steps.size(); step++) {
        uint64_t nGroups = 1 << starkInfo.starkStruct.steps[step].nBits;
        uint64_t groupSize = (1 << starkInfo.starkStruct.steps[step - 1].nBits) / nGroups;
        MerkleTreeGL *treeFRI =new MerkleTreeGL(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, nGroups, groupSize * FIELD_EXTENSION, NULL);
        for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
            bool res = treeFRI->verifyGroupProof(fproof.proof.fri.treesFRI[step - 1].root.data(), fproof.proof.fri.treesFRI[step - 1].polQueries[q][0].mp, friQueries[q], fproof.proof.fri.treesFRI[step - 1].polQueries[q][0].v);
            if(!res) {
                return false;
            }
        }
    }

    // TODO: Verify Query consistency

    for (uint64_t step=1; step < starkInfo.starkStruct.steps.size(); step++) {
        for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
            uint64_t idx = friQueries[q] % (1 << starkInfo.starkStruct.steps[step].nBits);     
            Goldilocks::Element value[3];
            FRI<Goldilocks::Element>::verify_fold(
                value,
                step, 
                starkInfo.starkStruct.nBitsExt, 
                starkInfo.starkStruct.steps[step].nBits, 
                starkInfo.starkStruct.steps[step - 1].nBits,
                &challenges[(starkInfo.challengesMap.size() + step)*FIELD_EXTENSION],
                idx,
                fproof.proof.fri.treesFRI[step - 1].polQueries[q][0].v
            );
            if (step < starkInfo.starkStruct.steps.size() - 1) {
                uint64_t groupIdx = idx / (1 << starkInfo.starkStruct.steps[step + 1].nBits);
                for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                    if(Goldilocks::toU64(value[i]) != Goldilocks::toU64(fproof.proof.fri.treesFRI[step].polQueries[q][0].v[groupIdx * FIELD_EXTENSION + i][0])) {
                        return false;
                    }
                }
            } else {
                for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                    if(Goldilocks::toU64(value[i]) != Goldilocks::toU64(fproof.proof.fri.pol[idx][i])) {
                        return false;
                    }
                }
            }
        }
    }
    cout << "TRUE" << endl;
    return true;
}



