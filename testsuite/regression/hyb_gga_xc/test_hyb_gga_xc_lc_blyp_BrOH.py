
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_blyp_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.092091176410612e+01, -2.092093556453805e+01, -2.092111889987638e+01, -2.092072761977528e+01, -2.092092367069668e+01, -2.092092367069668e+01, -3.345437662322988e+00, -3.345412006975568e+00, -3.344851100537296e+00, -3.346558795590669e+00, -3.345439354332128e+00, -3.345439354332128e+00, -5.379724894787213e-01, -5.376138216739705e-01, -5.287684046831537e-01, -5.336785057937010e-01, -5.378400612492961e-01, -5.378400612492961e-01, -6.478746020977054e-02, -6.720273549916886e-02, -6.627680117155056e-01, -2.091193139654693e-02, -6.552306968536405e-02, -6.552306968536405e-02, -1.754628908484422e-03, -1.836341104292725e-03, 1.685637926950095e-03, -8.518318972147146e-04, -1.817553605558866e-03, -1.817553605558866e-03, -4.922641722873854e+00, -4.922074410779058e+00, -4.922590010400407e+00, -4.922148791662153e+00, -4.922341851079069e+00, -4.922341851079069e+00, -1.943779234560595e+00, -1.954467381063848e+00, -1.943750750115517e+00, -1.952079908944292e+00, -1.951052858045368e+00, -1.951052858045368e+00, -4.349160263315638e-01, -4.696535878347577e-01, -4.068416703928082e-01, -4.197343262719825e-01, -4.547048926982898e-01, -4.547048926982898e-01, 3.638381901778757e-03, -5.714735611455318e-02, 1.936306161712255e-03, -1.688688813184341e+00, -7.981817097070389e-03, -7.981817097070389e-03, -8.221404327907885e-04, -9.378661766660042e-04, -7.055589201623547e-04, 5.644907198915359e-03, -8.556389464741196e-04, -8.556389464741196e-04, -4.361543090840874e-01, -4.364353039753845e-01, -4.363720627425104e-01, -4.362938012605184e-01, -4.363356188494929e-01, -4.363356188494929e-01, -4.178612823995949e-01, -3.662003975924076e-01, -3.826902000105999e-01, -3.968837480386627e-01, -3.897088250294052e-01, -3.897088250294052e-01, -4.989508249815175e-01, -1.015843921816328e-01, -1.449747123212746e-01, -2.178701644704646e-01, -1.799513382256905e-01, -1.799513382256904e-01, -3.178900439394869e-01, 1.249307755458565e-03, 6.221721870308797e-03, -1.982830696431332e-01, 3.546755410731412e-03, 3.546755410731453e-03, -2.050278068458789e-03, -2.597939173260485e-04, -4.914077904871480e-04, 2.106197739288628e-03, -7.286774944708301e-04, -7.286774944708290e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_blyp_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.512130362869325e+01, -2.512138557291490e+01, -2.512176621152852e+01, -2.512042405679513e+01, -2.512134671733331e+01, -2.512134671733331e+01, -3.975192294249463e+00, -3.975220953483364e+00, -3.976176257005226e+00, -3.975391261764890e+00, -3.975224328552954e+00, -3.975224328552954e+00, -6.880864132858389e-01, -6.873661930723900e-01, -6.702854029902823e-01, -6.758940570339540e-01, -6.878229823361491e-01, -6.878229823361491e-01, -1.361577783786838e-01, -1.377630570390159e-01, -8.417050888580722e-01, -9.529986580043834e-02, -1.366176322412709e-01, -1.366176322412709e-01, -2.328954765028890e-03, -2.437376316523872e-03, -1.181351280041791e-02, -1.131989368713482e-03, -2.412452115572261e-03, -2.412452115572261e-03, -6.105161298254333e+00, -6.107466849826983e+00, -6.105397744636093e+00, -6.107190412358667e+00, -6.106342203656367e+00, -6.106342203656367e+00, -2.149749409063189e+00, -2.166192799049806e+00, -2.142255241621332e+00, -2.155034733926150e+00, -2.170761754109290e+00, -2.170761754109290e+00, -5.861852484199677e-01, -6.516577721005388e-01, -5.509679003323622e-01, -5.819485430549792e-01, -6.130135002036944e-01, -6.130135002036944e-01, -6.695482654412133e-02, -1.545040713676543e-01, -6.354387116362120e-02, -2.218063873413044e+00, -7.707821421069368e-02, -7.707821421069368e-02, -1.092621337799150e-03, -1.246045426725468e-03, -9.380077574360690e-04, -2.686715493503054e-02, -1.137036894154814e-03, -1.137036894154814e-03, -6.162172759214100e-01, -6.097434574577436e-01, -6.119628320720029e-01, -6.137491646118061e-01, -6.128469514774442e-01, -6.128469514774442e-01, -5.930433432190878e-01, -4.959422212814269e-01, -5.208281083958438e-01, -5.461537576121590e-01, -5.329607109925119e-01, -5.329607109925120e-01, -6.885077125972485e-01, -2.033626002907288e-01, -2.440239490679945e-01, -3.190068350095906e-01, -2.776646765111878e-01, -2.776646765111877e-01, -4.386757078301466e-01, -9.657093496114667e-03, -2.169028876586142e-02, -2.928415131212821e-01, -4.393314085400769e-02, -4.393314085400760e-02, -2.720874758600241e-03, -3.459651905125609e-04, -6.537854799612925e-04, -3.888194827690625e-02, -9.686735708621689e-04, -9.686735708621675e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_blyp_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.139879861518144e-09, -7.139850140076842e-09, -7.139609895363895e-09, -7.140098759864727e-09, -7.139865087301544e-09, -7.139865087301544e-09, -8.565272954063057e-06, -8.565563799790330e-06, -8.572193692825639e-06, -8.553738849153772e-06, -8.565280967195651e-06, -8.565280967195651e-06, -2.736214415626883e-03, -2.730297929911728e-03, -2.538955875843178e-03, -2.503873366784924e-03, -2.734127873600157e-03, -2.734127873600157e-03, 2.156709931311501e-01, 1.999842371123699e-01, -1.770727646078714e-03, 8.771573849327871e-01, 2.108345496542266e-01, 2.108345496542266e-01, 3.277734942515662e-03, 6.556174806567940e-03, 1.009065383958130e+01, -2.362000587408641e-05, 5.627755735617702e-03, 5.627755735617702e-03, -2.041187143103534e-06, -2.042678052718131e-06, -2.041327958283490e-06, -2.042487344480720e-06, -2.041967000801198e-06, -2.041967000801198e-06, -5.801153519328598e-05, -5.700534235573174e-05, -5.790870131841044e-05, -5.712537387928226e-05, -5.746103993130511e-05, -5.746103993130511e-05, -5.519158835161166e-03, -5.759403412944293e-03, -6.273558010370479e-03, -7.155142471746201e-03, -5.149777362269265e-03, -5.149777362269265e-03, 2.434532476413930e+00, 2.298940975159181e-01, 2.525679918790525e+00, -1.106397338837769e-04, 1.547277215647680e+00, 1.547277215647680e+00, -2.229326717018219e-05, -3.211406591447861e-05, -2.997052591099870e-05, 7.276303149030202e+00, -3.250881264926589e-05, -3.250881264917999e-05, -7.596498511383072e-03, -6.995127820556795e-03, -7.180973876335893e-03, -7.344243378392445e-03, -7.260141002032245e-03, -7.260141002032245e-03, -8.584885532872306e-03, -6.559086239361812e-03, -7.001042357909225e-03, -7.427894332597921e-03, -7.217514315508711e-03, -7.217514315508712e-03, -4.892746982452810e-03, 6.850294597148893e-02, 1.652801100209051e-02, -9.822501846885066e-03, -1.659057369166675e-03, -1.659057369166626e-03, -7.977730661522955e-03, 1.017384779228779e+01, 8.462735758626598e+00, -1.336627586100867e-02, 4.247982075308005e+00, 4.247982075308005e+00, 2.870859882469796e-02, -1.298972077086140e-06, -6.998324692162112e-06, 4.741703702439668e+00, -2.701960185343823e-05, -2.701960185321660e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05