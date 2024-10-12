#ifndef EXPRESSIONS_CTX_HPP
#define EXPRESSIONS_CTX_HPP
#include "expressions_bin.hpp"
#include "const_pols.hpp"
#include "stark_info.hpp"
#include "steps.hpp"
#include "setup_ctx.hpp"

struct Params {
    ParserParams parserParams;
    uint64_t offset = false;
    bool inverse = 0;

    Params(ParserParams& params, bool inverse_ = false, uint64_t offset_ = 0) : parserParams(params), offset(offset_), inverse(inverse_) {}
};

struct Dest {
    Goldilocks::Element *dest = nullptr;
    std::vector<Params> params;

    Dest(Goldilocks::Element *dest_) : dest(dest_) {}

    void addParams(ParserParams& parserParams_, bool inverse_ = false, uint64_t offset_ = 0) {
        params.push_back(Params(parserParams_, inverse_, offset_));
    }
};

class ExpressionsCtx {
public:

    SetupCtx setupCtx;

    ExpressionsCtx(SetupCtx& _setupCtx) : setupCtx(_setupCtx) {};

    virtual ~ExpressionsCtx() {};
    
    virtual void calculateExpressions(StepsParams& params, ParserArgs &parserArgs, std::vector<Dest> dests, bool domainExtended) {};
 
    void calculateExpression(StepsParams& params, Goldilocks::Element* dest, uint64_t expressionId, bool inverse = false) {
        bool domainExtended = false;
         if(expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId) {
            setupCtx.expressionsBin.expressionsInfo[expressionId].destDim = 3;
            domainExtended = true;
        }
        Dest destStruct(dest);
        destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[expressionId], inverse);
        std::vector<Dest> dests = {destStruct};
        calculateExpressions(params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, domainExtended);
    }
};

#endif