#ifndef EXPRESSIONS_AVX512_HPP
#define EXPRESSIONS_AVX512_HPP
#include "expressions_ctx.hpp"

#ifdef __AVX512__

class ExpressionsAvx512 : public ExpressionsCtx {
public:
    uint64_t nrowsPack = 8;
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;
    ExpressionsAvx512(SetupCtx& setupCtx) : ExpressionsCtx(setupCtx) {};

    void setBufferTInfo(bool domainExtended, uint64_t expId) {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        offsetsStages.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);
        nColsStages.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);
        nColsStagesAcc.resize((setupCtx.starkInfo.nStages + 2)*nOpenings + 1);

        nCols = setupCtx.starkInfo.nConstants;
        uint64_t ns = setupCtx.starkInfo.nStages + 2;
        for(uint64_t o = 0; o < nOpenings; ++o) {
            for(uint64_t stage = 0; stage <= ns; ++stage) {
                std::string section = stage == 0 ? "const" : "cm" + to_string(stage);
                offsetsStages[(setupCtx.starkInfo.nStages + 2)*o + stage] = setupCtx.starkInfo.mapOffsets[std::make_pair(section, domainExtended)];
                nColsStages[(setupCtx.starkInfo.nStages + 2)*o + stage] = setupCtx.starkInfo.mapSectionsN[section];
                nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage] = stage == 0 && o == 0 ? 0 : nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage - 1] + nColsStages[stage - 1];
            }
        }

        nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings - 1] + nColsStages[(setupCtx.starkInfo.nStages + 2)*nOpenings - 1];
        if(expId == setupCtx.starkInfo.cExpId) {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + setupCtx.starkInfo.boundaries.size() + 1;
        } else if(expId == setupCtx.starkInfo.friExpId) {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + nOpenings*FIELD_EXTENSION;
        } else {
            nCols = nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + 1;
        }
    }

    inline void loadPolynomials(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> &dests, __m512i *bufferT_, uint64_t row, bool domainExtended) {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t domainSize = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;

        uint64_t extendBits = (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits);
        int64_t extend = domainExtended ? (1 << extendBits) : 1;
        uint64_t nextStrides[nOpenings];
        for(uint64_t i = 0; i < nOpenings; ++i) {
            uint64_t opening = setupCtx.starkInfo.openingPoints[i] < 0 ? setupCtx.starkInfo.openingPoints[i] + domainSize : setupCtx.starkInfo.openingPoints[i];
            nextStrides[i] = opening * extend;
        }

        Goldilocks::Element *constPols = domainExtended ? setupCtx.constPols.pConstPolsAddressExtended : setupCtx.constPols.pConstPolsAddress;

        std::vector<bool> constPolsUsed(setupCtx.starkInfo.constPolsMap.size(), false);
        std::vector<bool> cmPolsUsed(setupCtx.starkInfo.cmPolsMap.size(), false);

        for(uint64_t i = 0; i < dests.size(); ++i) {
            uint16_t* cmUsed = &parserArgs.cmPolsIds[dests[i].params[0].parserParams.cmPolsOffset];
            uint16_t* constUsed = &parserArgs.constPolsIds[dests[i].params[0].parserParams.constPolsOffset];

            for(uint64_t k = 0; k < dests[i].params[0].parserParams.nConstPolsUsed; ++k) {
                constPolsUsed[constUsed[k]] = true;
            }

            for(uint64_t k = 0; k < dests[i].params[0].parserParams.nCmPolsUsed; ++k) {
                cmPolsUsed[cmUsed[k]] = true;
            }
        }
        Goldilocks::Element bufferT[nOpenings*nrowsPack];

        for(uint64_t k = 0; k < constPolsUsed.size(); ++k) {
            if(!constPolsUsed[k]) continue;
            for(uint64_t o = 0; o < nOpenings; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT[nrowsPack*o + j] = constPols[l * nColsStages[0] + k];
                }
                Goldilocks::load_avx512(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o] + k], &bufferT[nrowsPack*o]);
            }
        }

        for(uint64_t k = 0; k < cmPolsUsed.size(); ++k) {
            if(!cmPolsUsed[k]) continue;
            PolMap polInfo = setupCtx.starkInfo.cmPolsMap[k];
            uint64_t stage = polInfo.stage;
            uint64_t stagePos = polInfo.stagePos;
            for(uint64_t d = 0; d < polInfo.dim; ++d) {
                for(uint64_t o = 0; o < nOpenings; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT[nrowsPack*o + j] = params.pols[offsetsStages[stage] + l * nColsStages[stage] + stagePos + d];
                    }
                    Goldilocks::load_avx512(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*o + stage] + (stagePos + d)], &bufferT[nrowsPack*o]);
                }
            }
        }

        if(dests[0].params[0].parserParams.expId == setupCtx.starkInfo.cExpId) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT[j] = setupCtx.constPols.x_2ns[row + j];
            }
            Goldilocks::load_avx512(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings]], &bufferT[0]);
            for(uint64_t d = 0; d < setupCtx.starkInfo.boundaries.size(); ++d) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    bufferT[j] = setupCtx.constPols.zi[row + j + d*domainSize];
                }
                Goldilocks::load_avx512(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + 1 + d], &bufferT[0]);
            }
        } else if(dests[0].params[0].parserParams.expId == setupCtx.starkInfo.friExpId) {
            for(uint64_t d = 0; d < setupCtx.starkInfo.openingPoints.size(); ++d) {
               for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        bufferT[j] = params.xDivXSub[(row + j + d*domainSize)*FIELD_EXTENSION + k];
                    }
                    Goldilocks::load_avx512(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings] + d*FIELD_EXTENSION + k], &bufferT[0]);
                }
            }
        } else {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                bufferT[j] = setupCtx.constPols.x_n[row + j];
            }
            Goldilocks::load_avx512(bufferT_[nColsStagesAcc[(setupCtx.starkInfo.nStages + 2)*nOpenings]], &bufferT[0]);
        }
    }

    inline void copyPolynomial(__m512i* destVals, bool inverse, uint64_t destId, uint64_t destDim, __m512i* tmp1, Goldilocks3::Element_avx512* tmp3) {
        if(destDim == 1) {
            if(inverse) {
                Goldilocks::Element buff[nrowsPack];
                Goldilocks::store_avx(buff, tmp1[destId]);
                // for(uint64_t j = 0; j < nrowsPack; ++j) {
                //     Goldilocks::inv(dests[i].dest[(row + j)*offset], dests[i].dest[(row + j)*offset]);
                // }
                Goldilocks::batchInverse(buff, buff, nrowsPack);
                Goldilocks::load_avx(destVals[0], buff);
            } else {
                Goldilocks::copy_avx512(destVals[0],tmp1[destId]);
            }
        } else if(destDim == 3) {
            if(inverse) {
                Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
                Goldilocks::store_avx(&buff[0], uint64_t(FIELD_EXTENSION), tmp3[destId][0]);
                Goldilocks::store_avx(&buff[1], uint64_t(FIELD_EXTENSION), tmp3[destId][1]);
                Goldilocks::store_avx(&buff[2], uint64_t(FIELD_EXTENSION), tmp3[destId][2]);
                Goldilocks3::batchInverse((Goldilocks3::Element *)buff, (Goldilocks3::Element *)buff, nrowsPack);
                Goldilocks::load_avx(destVals[0], &buff[0], uint64_t(FIELD_EXTENSION));
                Goldilocks::load_avx(destVals[1], &buff[1], uint64_t(FIELD_EXTENSION));
                Goldilocks::load_avx(destVals[2], &buff[2], uint64_t(FIELD_EXTENSION));
            } else {
                Goldilocks::copy_avx512(destVals[0], tmp3[destId][0]);
                Goldilocks::copy_avx512(destVals[1],tmp3[destId][1]);
                Goldilocks::copy_avx512(destVals[2],tmp3[destId][2]);
            }
        }
    }

    inline void storePolynomial(std::vector<Dest> dests, __m512i* destVals, uint64_t row) {
        for(uint64_t i = 0; i < dests.size(); ++i) {
            if(dests[i].params[0].parserParams.destDim == 1) {
                uint64_t offset = dests[i].params[0].offset != 0 ? dests[i].params[0].offset : 1;
                Goldilocks::store_avx512(&dests[i].dest[row*offset], uint64_t(offset), destVals[i*FIELD_EXTENSION]);
            } else {
                uint64_t offset = dests[i].params[0].offset != 0 ? dests[i].params[0].offset : FIELD_EXTENSION;
                Goldilocks::store_avx512(&dests[i].dest[row*offset], uint64_t(offset), destVals[i*FIELD_EXTENSION]);
                Goldilocks::store_avx512(&dests[i].dest[row*offset + 1], uint64_t(offset), destVals[i*FIELD_EXTENSION + 1]);
                Goldilocks::store_avx512(&dests[i].dest[row*offset + 2], uint64_t(offset), destVals[i*FIELD_EXTENSION + 2]);
            }
        }
    }

    inline void printTmp1(uint64_t row, __m512i tmp) {
        Goldilocks::Element buff[nrowsPack];
        Goldilocks::store_avx512(buff, tmp);
        for(uint64_t i = 0; i < 1; ++i) {
            cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
        }
    }

    inline void printTmp3(uint64_t row, Goldilocks3::Element_avx512 tmp) {
        Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
        Goldilocks::store_avx512(&buff[0], uint64_t(FIELD_EXTENSION), tmp[0]);
        Goldilocks::store_avx512(&buff[1], uint64_t(FIELD_EXTENSION), tmp[1]);
        Goldilocks::store_avx512(&buff[2], uint64_t(FIELD_EXTENSION), tmp[2]);
        for(uint64_t i = 0; i < 1; ++i) {
            cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
        }
    }

    inline void printCommit(uint64_t row, __m512i* bufferT, bool extended) {
        if(extended) {
            Goldilocks::Element buff[FIELD_EXTENSION*nrowsPack];
            Goldilocks::store_avx512(&buff[0], uint64_t(FIELD_EXTENSION), bufferT[0]);
            Goldilocks::store_avx512(&buff[1], uint64_t(FIELD_EXTENSION), bufferT[setupCtx.starkInfo.openingPoints.size()]);
            Goldilocks::store_avx512(&buff[2], uint64_t(FIELD_EXTENSION), bufferT[2*setupCtx.starkInfo.openingPoints.size()]);
            for(uint64_t i = 0; i < 1; ++i) {
                cout << "Value at row " << row + i << " is [" << Goldilocks::toString(buff[FIELD_EXTENSION*i]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 1]) << ", " << Goldilocks::toString(buff[FIELD_EXTENSION*i + 2]) << "]" << endl;
            }
        } else {
            Goldilocks::Element buff[nrowsPack];
            Goldilocks::store_avx512(&buff[0], bufferT[0]);
            for(uint64_t i = 0; i < 1; ++i) {
                cout << "Value at row " << row + i << " is " << Goldilocks::toString(buff[i]) << endl;
            }
        }
    }

    void calculateExpressions(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> dests, bool domainExtended) override {
        uint64_t nOpenings = setupCtx.starkInfo.openingPoints.size();
        uint64_t domainSize = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;

        setBufferTInfo(domainExtended, dests[0].params[0].parserParams.expId);

        Goldilocks3::Element_avx512 challenges[setupCtx.starkInfo.challengesMap.size()];
        for(uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); ++i) {
            challenges[i][0] = _mm512_set1_epi64(params.challenges[i * FIELD_EXTENSION].fe);
            challenges[i][1] = _mm512_set1_epi64(params.challenges[i * FIELD_EXTENSION + 1].fe);
            challenges[i][2] = _mm512_set1_epi64(params.challenges[i * FIELD_EXTENSION + 2].fe);

        }

        __m512i** numbers_ = new __m512i*[dests.size()];
        for(uint64_t i = 0; i < dests.size(); ++i) {
            uint64_t* numbers = &parserArgs.numbers[dests[i].params[0].parserParams.numbersOffset];
            numbers_[i] = new __m256i[dests[i].params[0].parserParams.nNumbers];
            for(uint64_t j = 0; j < dests[i].params[0].parserParams.nNumbers; ++j) {
                numbers_[i][j] = _mm512_set1_epi64(numbers[j]);
            }
        }

        __m512i publics[setupCtx.starkInfo.nPublics];
        for(uint64_t i = 0; i < setupCtx.starkInfo.nPublics; ++i) {
            publics[i] = _mm512_set1_epi64(params.publicInputs[i].fe);
        }

        Goldilocks3::Element_avx512 subproofValues[setupCtx.starkInfo.nSubProofValues];
        for(uint64_t i = 0; i < setupCtx.starkInfo.nSubProofValues; ++i) {
            subproofValues[i][0] = _mm512_set1_epi64(params.subproofValues[i * FIELD_EXTENSION].fe);
            subproofValues[i][1] = _mm512_set1_epi64(params.subproofValues[i * FIELD_EXTENSION + 1].fe);
            subproofValues[i][2] = _mm512_set1_epi64(params.subproofValues[i * FIELD_EXTENSION + 2].fe);
        }

        Goldilocks3::Element_avx512 evals[setupCtx.starkInfo.evMap.size()];
        for(uint64_t i = 0; i < setupCtx.starkInfo.evMap.size(); ++i) {
            evals[i][0] = _mm512_set1_epi64(params.evals[i * FIELD_EXTENSION].fe);
            evals[i][1] = _mm512_set1_epi64(params.evals[i * FIELD_EXTENSION + 1].fe);
            evals[i][2] = _mm512_set1_epi64(params.evals[i * FIELD_EXTENSION + 2].fe);
        }

    #pragma omp parallel for
        for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {
            __m512i bufferT_[nOpenings*nCols];

            loadPolynomials(params, parserArgs, dests, bufferT_, i, domainExtended);

            __m512i destVals[dests.size()*FIELD_EXTENSION];

            for(uint64_t j = 0; j < dests.size(); ++j) {
                uint64_t i_args = 0;

                uint8_t* ops = &parserArgs.ops[dests[j].params[0].parserParams.opsOffset];
                uint16_t* args = &parserArgs.args[dests[j].params[0].parserParams.argsOffset];
                __m512i tmp1[dests[j].params[0].parserParams.nTemp1];
                Goldilocks3::Element_avx512 tmp3[dests[j].params[0].parserParams.nTemp3];

                for (uint64_t kk = 0; kk < dests[j].params[0].parserParams.nOps; ++kk) {
                    switch (ops[kk]) {
                        case 0: {
                           // COPY commit1 to tmp1
                            Goldilocks::copy_avx512(tmp1[args[i_args]], bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                            i_args += 3;
                            break;
                        }
                        case 1: {
                            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                            i_args += 6;
                            break;
                        }
                        case 2: {
                            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 3: {
                            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 4: {
                            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[j][args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 5: {
                           // COPY tmp1 to tmp1
                            Goldilocks::copy_avx512(tmp1[args[i_args]], tmp1[args[i_args + 1]]);
                            i_args += 2;
                            break;
                        }
                        case 6: {
                            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], tmp1[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 7: {
                            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], publics[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 8: {
                            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], tmp1[args[i_args + 2]], numbers_[j][args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 9: {
                           // COPY public to tmp1
                            Goldilocks::copy_avx512(tmp1[args[i_args]], publics[args[i_args + 1]]);
                            i_args += 2;
                            break;
                        }
                        case 10: {
                            // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], publics[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 11: {
                            // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], publics[args[i_args + 2]], numbers_[j][args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 12: {
                           // COPY number to tmp1
                            Goldilocks::copy_avx512(tmp1[args[i_args]], numbers_[j][args[i_args + 1]]);
                            i_args += 2;
                            break;
                        }
                        case 13: {
                            // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                            Goldilocks::op_avx512(args[i_args], tmp1[args[i_args + 1]], numbers_[j][args[i_args + 2]], numbers_[j][args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 14: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                            i_args += 6;
                            break;
                        }
                        case 15: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp1[args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 16: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], publics[args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 17: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], numbers_[j][args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 18: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 19: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp1[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 20: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], publics[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 21: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], numbers_[j][args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 22: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 23: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], tmp1[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 24: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], publics[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 25: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], numbers_[j][args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 26: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: commit1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 27: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: tmp1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], tmp1[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 28: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: public
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], publics[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 29: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: number
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], numbers_[j][args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 30: {
                           // COPY commit3 to tmp3
                            Goldilocks3::copy_avx512(tmp3[args[i_args]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]]);
                            i_args += 3;
                            break;
                        }
                        case 31: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]]);
                            i_args += 6;
                            break;
                        }
                        case 32: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], tmp3[args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 33: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], challenges[args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 34: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: subproofValue
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], subproofValues[args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 35: {
                           // COPY tmp3 to tmp3
                            Goldilocks3::copy_avx512(tmp3[args[i_args]], tmp3[args[i_args + 1]]);
                            i_args += 2;
                            break;
                        }
                        case 36: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], tmp3[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 37: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], challenges[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 38: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: subproofValue
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], subproofValues[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 39: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], challenges[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 40: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: subproofValue
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], subproofValues[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 41: {
                            // OPERATION WITH DEST: tmp3 - SRC0: subproofValue - SRC1: subproofValue
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], subproofValues[args[i_args + 2]], subproofValues[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 42: {
                           // COPY eval to tmp3
                            Goldilocks3::copy_avx512(tmp3[args[i_args]], evals[args[i_args + 1]]);
                            i_args += 2;
                            break;
                        }
                        case 43: {
                            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], challenges[args[i_args + 2]], evals[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 44: {
                            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], tmp3[args[i_args + 2]], evals[args[i_args + 3]]);
                            i_args += 4;
                            break;
                        }
                        case 45: {
                            // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                            Goldilocks3::op_31_avx512(args[i_args], tmp3[args[i_args + 1]], evals[args[i_args + 2]], bufferT_[nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        case 46: {
                            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                            Goldilocks3::op_avx512(args[i_args], tmp3[args[i_args + 1]], (Goldilocks3::Element_avx512 &)bufferT_[nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]], evals[args[i_args + 4]]);
                            i_args += 5;
                            break;
                        }
                        default: {
                            std::cout << " Wrong operation!" << std::endl;
                            exit(1);
                        }
                    }
                }

                if (i_args != dests[j].params[0].parserParams.nArgs) std::cout << " " << i_args << " - " << dests[j].params[0].parserParams.nArgs << std::endl;
                assert(i_args == dests[j].params[0].parserParams.nArgs);

                copyPolynomial(&destVals[j*FIELD_EXTENSION], dests[j].params[0].inverse, dests[j].params[0].parserParams.destId, dests[j].params[0].parserParams.destDim, tmp1, tmp3);
            }
            storePolynomial(dests, destVals, i);

        }
        for (uint64_t i = 0; i < dests.size(); ++i) {
            delete numbers_[i];
        }

        delete[] numbers_;
    }
};

#endif
#endif