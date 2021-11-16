# Reporting vulnerabilities

Please email reports about any security related issues you find to
`EMAIL`. For critical problems, you may encrypt your report (see
below).

Please use a descriptive subject line for your report email. After the initial
reply to your report, the security team will endeavor to keep you informed of
the progress being made towards a fix and announcement.

In addition, please include the following information along with your report:

* Your name and affiliation (if any).
* A description of the technical details of the vulnerabilities. It is very
  important to let us know how we can reproduce your findings.
* An explanation who can exploit this vulnerability, and what they gain when
  doing so -- write an attack scenario. This will help us evaluate your report
  quickly, especially if the issue is complex.
* Whether this vulnerability public or known to third parties. If it is, please
  provide details.

If you believe that an existing (public) issue is security-related, please send
an email to `EMAIL`. The email should include the issue ID and
a short description of why it should be handled according to this security
policy.

Once an issue is reported, PADL uses the following disclosure process:

* When a report is received, we confirm the issue and determine its severity.
* If we know of specific third-party services or software based on PADL
  that require mitigation before publication, those projects will be notified.
* An advisory is prepared (but not published) which details the problem and
  steps for mitigation.
* The vulnerability is fixed and potential workarounds are identified.
* Wherever possible, the fix is also prepared for the branches corresponding to
  all releases of PADL at most one year old. We will attempt to commit
  these fixes as soon as possible, and as close together as possible.

## Encryption key for `EMAIL`

If your disclosure is extremely sensitive, you may choose to encrypt your
report using the key below. Please only use this for critical security
reports.

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
# TODO
-----END PGP PUBLIC KEY BLOCK-----
```
